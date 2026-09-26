//! Visualization and Graphviz rendering utilities for `GraphEnv` state-transition systems.

use std::collections::{BTreeSet, HashMap};
use std::io;

use graphviz_rust::cmd::{CommandArg, Format};
use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::exec;
use graphviz_rust::print;
use graphviz_rust::printer::PrinterContext;

use mcts_traits::GraphEnv;

use crate::builder::escape_dot_string;

/// Converts a [`GraphEnv`] into a Graphviz [`Graph`] abstract syntax tree.
pub fn graph_env_to_graph<const N: usize>(
    env: &GraphEnv<N>,
    title: &str,
    action_names: Option<&HashMap<u32, String>>,
) -> Graph {
    let mut states = BTreeSet::new();
    states.insert(env.initial_state);
    for &s in env.legal_actions.keys() {
        states.insert(s);
    }
    for ((from, _), trans) in &env.transitions {
        states.insert(*from);
        states.insert(trans.next_state);
    }
    for &term in &env.terminal_states {
        states.insert(term);
    }

    let mut stmts: Vec<Stmt> = Vec::new();
    stmts.push(Stmt::GAttribute(GraphAttributes::Graph(vec![
        attr!("rankdir", esc "LR"),
        attr!("bgcolor", "\"#ffffff\""),
        attr!("nodesep", "\"0.5\""),
        attr!("ranksep", "\"0.7\""),
        attr!("splines", "\"spline\""),
        attr!("fontname", esc "Helvetica,Arial,sans-serif"),
    ])));
    stmts.push(Stmt::GAttribute(GraphAttributes::Node(vec![
        attr!("fontname", esc "Helvetica,Arial,sans-serif"),
        attr!("fontsize", "\"11\""),
    ])));
    stmts.push(Stmt::GAttribute(GraphAttributes::Edge(vec![
        attr!("fontname", esc "Helvetica,Arial,sans-serif"),
        attr!("fontsize", "\"10\""),
    ])));

    for &s in &states {
        let node_name = format!("s_{s}");
        let is_initial = s == env.initial_state;
        let is_term = env.is_terminal(&s);

        let (header_bg, border_color, title_text, subtitle) = if is_term {
            ("#059669", "#059669", format!("S_{s}"), "Terminal")
        } else if is_initial {
            ("#1D4ED8", "#1D4ED8", format!("S_{s}"), "Initial")
        } else {
            ("#4B5563", "#4B5563", format!("S_{s}"), "State")
        };

        let penwidth = if is_initial || is_term {
            "\"2.0\""
        } else {
            "\"1.2\""
        };
        let style = "\"rounded,filled\"";

        let html_label = format!(
            "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"4\" BGCOLOR=\"#FFFFFF\">\
            <TR><TD BGCOLOR=\"{header_bg}\" ALIGN=\"CENTER\"><FONT COLOR=\"#FFFFFF\"><B>{title_text}</B></FONT></TD></TR>\
            <TR><TD ALIGN=\"CENTER\" BGCOLOR=\"#F8FAFC\"><FONT POINT-SIZE=\"9\" COLOR=\"#334155\">{subtitle}</FONT></TD></TR>\
            </TABLE>>"
        );

        stmts.push(Stmt::Node(Node::new(
            node_id!(node_name),
            vec![
                attr!("label", html_label),
                attr!("shape", "\"box\""),
                attr!("style", style),
                attr!("fillcolor", "\"#FFFFFF\""),
                attr!("color", esc border_color),
                attr!("penwidth", penwidth),
            ],
        )));
    }

    // Deterministic ordering of transitions
    let mut sorted_transitions: Vec<_> = env.transitions.iter().collect();
    sorted_transitions.sort_by_key(|((from, a), trans)| (*from, *a, trans.next_state));

    for (&(from, action), trans) in sorted_transitions {
        let from_name = format!("s_{from}");
        let to_name = format!("s_{}", trans.next_state);

        let action_str = if let Some(map) = action_names {
            map.get(&action)
                .cloned()
                .unwrap_or_else(|| format!("a_{action}"))
        } else {
            format!("a_{action}")
        };

        let reward_str = if N == 1 {
            format!("R: {:+.1}", trans.reward[0])
        } else {
            format!("R: {:?}", trans.reward)
        };

        let label = format!("{action_str}\n{reward_str}");
        let escaped_label = escape_dot_string(&label);

        let edge_stmt = edge!(node_id!(from_name) => node_id!(to_name);
            attr!("label", esc &escaped_label),
            attr!("color", esc "#4B5563"),
            attr!("fontcolor", esc "#1F2937"),
            attr!("penwidth", "\"1.4\""),
            attr!("arrowhead", "\"vee\"")
        );
        stmts.push(Stmt::Edge(edge_stmt));
    }

    Graph::DiGraph {
        id: Id::Plain(title.to_string()),
        strict: false,
        stmts,
    }
}

/// Generates a Graphviz DOT string representation of a [`GraphEnv`].
pub fn graph_env_to_dot<const N: usize>(
    env: &GraphEnv<N>,
    title: &str,
    action_names: Option<&HashMap<u32, String>>,
) -> String {
    let graph = graph_env_to_graph(env, title, action_names);
    print(graph, &mut PrinterContext::default())
}

/// Renders a [`GraphEnv`] into an SVG string using the local Graphviz `dot` toolchain.
///
/// # Errors
///
/// Returns an [`io::Error`] if `dot` execution fails or output cannot be decoded.
pub fn render_graph_env_svg<const N: usize>(
    env: &GraphEnv<N>,
    title: &str,
    action_names: Option<&HashMap<u32, String>>,
) -> io::Result<String> {
    let graph = graph_env_to_graph(env, title, action_names);
    let bytes = exec(
        graph,
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
