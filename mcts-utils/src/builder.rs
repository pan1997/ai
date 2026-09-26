//! Core Graphviz graph builder for `TreeStore`.

use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;

use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;

use mcts_engine::tree_store::{
    EdgeId, EdgeStatsStore, NodeId as MctsNodeId, NodeStatus, TreeStore,
};
use mcts_traits::AgentId;

use crate::options::{NodeFilter, TreeVisualizerOptions};
use crate::stats::EdgeStatsView;

/// Builds a Graphviz [`Graph`] representation from an MCTS [`TreeStore`].
pub fn build_dot_graph<Action, Reward, Stats, StepDelta>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: MctsNodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
) -> Graph
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    assert!(
        root.as_usize() < tree.num_nodes(),
        "build_dot_graph: root NodeId ({:?}) out of bounds (total nodes: {})",
        root,
        tree.num_nodes()
    );

    // 1. Calculate incoming visit counts for all nodes in the tree
    let mut node_visits = vec![0u32; tree.num_nodes()];
    node_visits[root.as_usize()] = tree.child_edges(root).map(|e| tree.stats.visits(e)).sum();

    for n in 0..tree.num_nodes() {
        let node = MctsNodeId(n as u32);
        if node == root {
            continue;
        }
        let pe = tree.parent_edge(node);
        if !pe.is_valid() {
            continue;
        }
        let pe_visits = tree.stats.visits(pe);
        if pe_visits == 0 {
            node_visits[n] = 0;
            continue;
        }

        match tree.node_status(node) {
            NodeStatus::Expanded => {
                let child_sum: u32 = tree.child_edges(node).map(|e| tree.stats.visits(e)).sum();
                node_visits[n] = (1 + child_sum).min(pe_visits);
            }
            NodeStatus::Terminal | NodeStatus::Unexpanded => {
                let delta_count = tree.delta_children(pe).count();
                if delta_count <= 1 {
                    node_visits[n] = pe_visits;
                } else {
                    node_visits[n] = 1;
                }
            }
        }
    }

    // 2. Determine allowed nodes based on NodeFilter
    let mut allowed_nodes: HashSet<MctsNodeId> = HashSet::new();
    allowed_nodes.insert(root); // Root is unconditionally included

    match &options.node_filter {
        NodeFilter::All => {
            for n in 0..tree.num_nodes() {
                allowed_nodes.insert(MctsNodeId(n as u32));
            }
        }
        NodeFilter::MinVisits(min_v) => {
            for (n, &visits) in node_visits.iter().enumerate() {
                if visits >= *min_v {
                    allowed_nodes.insert(MctsNodeId(n as u32));
                }
            }
        }
        NodeFilter::TopNodes(top_n) => {
            let mut indices: Vec<usize> = (0..tree.num_nodes()).collect();
            indices.sort_by(|&a, &b| node_visits[b].cmp(&node_visits[a]));
            for &idx in indices.iter().take(*top_n) {
                allowed_nodes.insert(MctsNodeId(idx as u32));
            }
        }
        NodeFilter::Custom(pred) => {
            for (n, &visits) in node_visits.iter().enumerate() {
                let node = MctsNodeId(n as u32);
                if pred(node, visits) {
                    allowed_nodes.insert(node);
                }
            }
        }
    }

    // 3. Identify Principal Variation (PV) edges if highlighting is requested
    let mut pv_edges = HashSet::new();
    let mut pv_steps = std::collections::HashMap::new();
    if options.highlight_pv {
        let mut curr = root;
        while tree.node_status(curr) == NodeStatus::Expanded {
            let mut best_edge = EdgeId::INVALID;
            let mut max_visits = 0u32;
            for edge in tree.child_edges(curr) {
                let v = tree.stats.visits(edge);
                if v > max_visits {
                    max_visits = v;
                    best_edge = edge;
                }
            }
            if best_edge.is_valid() && max_visits > 0 {
                pv_edges.insert(best_edge);
                let mut best_child = tree.edge_child(best_edge);
                let mut best_child_visits = if best_child.is_valid() {
                    node_visits[best_child.as_usize()]
                } else {
                    0
                };
                for (_delta, c_node) in tree.delta_children(best_edge) {
                    if c_node.is_valid() && node_visits[c_node.as_usize()] > best_child_visits {
                        best_child_visits = node_visits[c_node.as_usize()];
                        best_child = c_node;
                    }
                }
                if best_child.is_valid() && best_child_visits > 0 {
                    pv_steps.insert(best_edge, best_child);
                    curr = best_child;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    // 4. Find maximum edge visits for proportional edge width scaling
    let mut max_edge_visits = 1u32;
    for e in 0..tree.num_edges() {
        let v = tree.stats.visits(EdgeId(e as u32));
        if v > max_edge_visits {
            max_edge_visits = v;
        }
    }

    // 5. BFS Traversal building nodes and edges
    let mut visited_nodes = HashSet::new();
    let mut statements: Vec<Stmt> = Vec::new();

    // Default Graph Attributes
    statements.push(Stmt::GAttribute(GraphAttributes::Graph(vec![
        attr!("rankdir", esc options.rankdir.as_str()),
        attr!("bgcolor", "\"#ffffff\""),
        attr!("nodesep", "\"0.4\""),
        attr!("ranksep", "\"0.6\""),
        attr!("splines", "\"spline\""),
        attr!("fontname", esc & options.font_name),
    ])));
    statements.push(Stmt::GAttribute(GraphAttributes::Node(vec![
        attr!("fontname", esc & options.font_name),
        attr!("fontsize", "\"11\""),
    ])));
    statements.push(Stmt::GAttribute(GraphAttributes::Edge(vec![
        attr!("fontname", esc & options.font_name),
        attr!("fontsize", "\"10\""),
    ])));

    let mut queue = VecDeque::new();
    queue.push_back((root, 0usize));
    visited_nodes.insert(root);

    while let Some((node, depth)) = queue.pop_front() {
        let status = tree.node_status(node);
        let agent = tree.node_agent(node);
        let visits = node_visits[node.as_usize()];

        // Render Node Statement
        let node_stmt = render_node(node, status, agent, node == root, visits, options);
        statements.push(Stmt::Node(node_stmt));

        // If max depth reached or node not expanded, do not traverse children
        if let Some(limit) = options.max_depth
            && depth >= limit
        {
            continue;
        }
        if status != NodeStatus::Expanded {
            continue;
        }

        // Traverse child edges
        for edge in tree.child_edges(node) {
            let edge_visits = tree.stats.visits(edge);
            if edge_visits < options.min_visits {
                continue;
            }

            let is_pv = pv_edges.contains(&edge);

            // Format Action label
            let action_label = if let Some(fmt) = &options.action_formatter {
                fmt(tree.edge_action(edge))
            } else {
                let raw = format!("{:?}", tree.edge_action(edge));
                raw.strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .unwrap_or(&raw)
                    .to_string()
            };

            // Format edge statistics
            let stats_label = tree.stats.format_stats(edge);

            // Optional transition reward
            let reward_label = if options.show_rewards {
                tree.edge_reward(edge)
                    .map(|r| format!("\nR: {:?}", r))
                    .unwrap_or_default()
            } else {
                String::new()
            };

            // Check stochastic delta branches
            let delta_branches: Vec<(&StepDelta, MctsNodeId)> = tree.delta_children(edge).collect();

            let is_stochastic = delta_branches.len() > 1
                || (delta_branches.len() == 1 && *delta_branches[0].0 != StepDelta::default());

            let pv_child = pv_steps.get(&edge).copied();

            if options.render_chance_nodes && !delta_branches.is_empty() {
                let valid_branches: Vec<_> = delta_branches
                    .into_iter()
                    .filter(|(_, child)| child.is_valid() && allowed_nodes.contains(child))
                    .collect();

                if !valid_branches.is_empty() {
                    let chance_node_name = format!("c_{}_{}", node.0, edge.0);
                    let chance_node = Node::new(
                        node_id!(chance_node_name.clone()),
                        vec![
                            attr!("label", "\"\""),
                            attr!("shape", "\"circle\""),
                            attr!("style", "\"filled\""),
                            attr!("fillcolor", "\"#FB923C\""),
                            attr!("color", "\"#EA580C\""),
                            attr!("penwidth", "\"1.5\""),
                            attr!("width", "\"0.22\""),
                            attr!("height", "\"0.22\""),
                            attr!("fixedsize", "\"true\""),
                            attr!(
                                "tooltip",
                                esc & format!("Chance outcome for {action_label}")
                            ),
                        ],
                    );
                    statements.push(Stmt::Node(chance_node));

                    let action_stats_label = if stats_label.is_empty() && reward_label.is_empty() {
                        action_label.clone()
                    } else if stats_label.is_empty() {
                        format!("{action_label}{reward_label}")
                    } else {
                        format!("{action_label}\n{stats_label}{reward_label}")
                    };

                    let in_edge = render_named_edge(
                        &format!("n_{}", node.0),
                        &chance_node_name,
                        &action_stats_label,
                        edge_visits,
                        max_edge_visits,
                        is_pv,
                        false,
                    );
                    statements.push(Stmt::Edge(in_edge));

                    for (delta, child_node) in valid_branches {
                        let delta_label = if let Some(fmt) = &options.delta_formatter {
                            fmt(delta)
                        } else {
                            let raw_delta = format!("{:?}", delta);
                            raw_delta
                                .strip_prefix('"')
                                .and_then(|s| s.strip_suffix('"'))
                                .unwrap_or(&raw_delta)
                                .to_string()
                        };

                        let branch_label = format!("Δ: {delta_label}");
                        let is_branch_pv = is_pv && pv_child == Some(child_node);
                        let out_edge = render_named_edge(
                            &chance_node_name,
                            &format!("n_{}", child_node.0),
                            &branch_label,
                            edge_visits,
                            max_edge_visits,
                            is_branch_pv,
                            true,
                        );
                        statements.push(Stmt::Edge(out_edge));

                        if visited_nodes.insert(child_node) {
                            queue.push_back((child_node, depth + 1));
                        }
                    }
                }
            } else if is_stochastic {
                for (delta, child_node) in delta_branches {
                    if child_node.is_valid() && allowed_nodes.contains(&child_node) {
                        let delta_label = if let Some(fmt) = &options.delta_formatter {
                            fmt(delta)
                        } else {
                            let raw_delta = format!("{:?}", delta);
                            raw_delta
                                .strip_prefix('"')
                                .and_then(|s| s.strip_suffix('"'))
                                .unwrap_or(&raw_delta)
                                .to_string()
                        };

                        let full_edge_label = if stats_label.is_empty() && reward_label.is_empty() {
                            format!("{action_label}\nΔ: {delta_label}")
                        } else {
                            format!("{action_label}\nΔ: {delta_label}\n{stats_label}{reward_label}")
                        };
                        let is_branch_pv = is_pv && pv_child == Some(child_node);
                        let edge_stmt = render_edge(
                            node,
                            child_node,
                            &full_edge_label,
                            edge_visits,
                            max_edge_visits,
                            is_branch_pv,
                        );
                        statements.push(Stmt::Edge(edge_stmt));

                        if visited_nodes.insert(child_node) {
                            queue.push_back((child_node, depth + 1));
                        }
                    }
                }
            } else {
                let child_node = tree.edge_child(edge);
                if child_node.is_valid() && allowed_nodes.contains(&child_node) {
                    let full_edge_label = if stats_label.is_empty() && reward_label.is_empty() {
                        action_label
                    } else if stats_label.is_empty() {
                        format!("{action_label}{reward_label}")
                    } else {
                        format!("{action_label}\n{stats_label}{reward_label}")
                    };

                    let edge_stmt = render_edge(
                        node,
                        child_node,
                        &full_edge_label,
                        edge_visits,
                        max_edge_visits,
                        is_pv,
                    );
                    statements.push(Stmt::Edge(edge_stmt));

                    if visited_nodes.insert(child_node) {
                        queue.push_back((child_node, depth + 1));
                    }
                }
            }
        }
    }

    Graph::DiGraph {
        id: Id::Plain(options.graph_id.clone()),
        strict: false,
        stmts: statements,
    }
}

/// Helper to render a styled node into a [`Node`] struct.
fn render_node<Action, StepDelta>(
    node: MctsNodeId,
    status: NodeStatus,
    agent: AgentId,
    is_root: bool,
    incoming_visits: u32,
    options: &TreeVisualizerOptions<Action, StepDelta>,
) -> Node {
    let node_name = format!("n_{}", node.0);

    // Custom node label override if provided
    if let Some(custom_fn) = &options.custom_node_label
        && let Some(custom_label) = custom_fn(node, status, agent)
    {
        let escaped_custom = escape_dot_string(&custom_label);
        return Node::new(
            node_id!(node_name),
            vec![
                attr!("label", esc & escaped_custom),
                attr!("shape", "\"box\""),
                attr!("style", "\"rounded,filled\""),
                attr!("fillcolor", "\"#f8f9fa\""),
                attr!("color", "\"#495057\""),
            ],
        );
    }

    // Color palette based on status and agent
    let (header_bg, border_color, title, subtitle) = if is_root {
        (
            "#1D4ED8",
            "#1D4ED8",
            format!("Root (Agent {})", agent.0),
            format!("Visits: {incoming_visits}"),
        )
    } else {
        match status {
            NodeStatus::Terminal => (
                "#059669",
                "#059669",
                format!("Node #{} (Terminal)", node.0),
                format!("Visits: {incoming_visits}"),
            ),
            NodeStatus::Expanded => {
                let color = if agent.0 == 0 { "#2563EB" } else { "#D97706" };
                (
                    color,
                    "#4B5563",
                    format!("Node #{} (Agent {})", node.0, agent.0),
                    format!("Visits: {incoming_visits}"),
                )
            }
            NodeStatus::Unexpanded => (
                "#6B7280",
                "#9CA3AF",
                format!("Node #{} (Unexpanded)", node.0),
                format!("Visits: {incoming_visits}"),
            ),
        }
    };

    let style = if status == NodeStatus::Unexpanded {
        "\"rounded,filled,dashed\""
    } else {
        "\"rounded,filled\""
    };

    let penwidth = if is_root || status == NodeStatus::Terminal {
        "\"2.0\""
    } else {
        "\"1.2\""
    };

    // HTML table label
    let html_label = format!(
        "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"4\" BGCOLOR=\"#FFFFFF\">\
        <TR><TD BGCOLOR=\"{header_bg}\" ALIGN=\"CENTER\"><FONT COLOR=\"#FFFFFF\"><B>{title}</B></FONT></TD></TR>\
        <TR><TD ALIGN=\"CENTER\" BGCOLOR=\"#F8FAFC\"><FONT POINT-SIZE=\"10\" COLOR=\"#334155\">{subtitle}</FONT></TD></TR>\
        </TABLE>>"
    );

    Node::new(
        node_id!(node_name),
        vec![
            attr!("label", html_label),
            attr!("shape", "\"box\""),
            attr!("style", style),
            attr!("fillcolor", "\"#FFFFFF\""),
            attr!("color", esc border_color),
            attr!("penwidth", penwidth),
        ],
    )
}

/// Helper to render a directed edge with explicit string node IDs into an [`Edge`] struct.
fn render_named_edge(
    from_name: &str,
    to_name: &str,
    label: &str,
    visits: u32,
    max_visits: u32,
    is_pv: bool,
    dashed: bool,
) -> Edge {
    let (color, fontcolor, penwidth, style) = if is_pv {
        (
            "#2563EB",
            "#1D4ED8",
            "\"2.8\"".to_string(),
            if dashed { "\"dashed\"" } else { "\"solid\"" },
        )
    } else if visits > 0 {
        let width = 1.0 + 2.0 * (visits as f32 / max_visits as f32).min(1.0);
        (
            "#4B5563",
            "#1F2937",
            format!("\"{:.1}\"", width),
            if dashed { "\"dashed\"" } else { "\"solid\"" },
        )
    } else {
        ("#94A3B8", "#94A3B8", "\"1.0\"".to_string(), "\"dashed\"")
    };

    let escaped_label = escape_dot_string(label);

    edge!(node_id!(from_name) => node_id!(to_name);
        attr!("label", esc &escaped_label),
        attr!("color", esc color),
        attr!("fontcolor", esc fontcolor),
        attr!("penwidth", penwidth),
        attr!("style", style),
        attr!("arrowhead", "\"vee\"")
    )
}

/// Helper to render a directed edge into an [`Edge`] struct.
fn render_edge(
    from: MctsNodeId,
    to: MctsNodeId,
    label: &str,
    visits: u32,
    max_visits: u32,
    is_pv: bool,
) -> Edge {
    render_named_edge(
        &format!("n_{}", from.0),
        &format!("n_{}", to.0),
        label,
        visits,
        max_visits,
        is_pv,
        false,
    )
}

/// Escapes a string for a Graphviz DOT string literal (handles quotes, newlines, backslashes).
#[inline]
pub(crate) fn escape_dot_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            _ => out.push(c),
        }
    }
    out
}
