//! Unit tests and artificial search tree validation.

use mcts_engine::selection::MultiAgentPuctStats;
use mcts_engine::selection::ismcts::IsmctsStats;
use mcts_engine::tree_store::{EdgeId, TreeStore};
use mcts_traits::AgentId;

use crate::options::{RankDir, TreeVisualizerOptions};
use crate::render::{render_png, render_svg, to_dot};

#[test]
fn test_artificial_2_node_tree() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<&'static str, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    let e0 = tree.expand_node(root, &["StepRight"]);
    let _child = tree.insert_node(e0, AgentId(1));

    tree.stats.visits[0] = 42;
    tree.stats.priors[0] = 1.0;
    tree.stats.mean_value[0] = [0.75];

    let options = TreeVisualizerOptions::new().with_rankdir(RankDir::TopToBottom);
    let dot = to_dot(&tree, root, &options);

    println!("--- 2-Node Tree DOT ---\n{dot}");

    // Validate that graphviz-rust can parse the generated DOT back into AST
    let parsed = graphviz_rust::parse(&dot);
    assert!(
        parsed.is_ok(),
        "Generated DOT must be syntactically valid: {:?}",
        parsed.err()
    );

    // Verify key labels
    assert!(dot.contains("n_0"));
    assert!(dot.contains("n_1"));
    assert!(dot.contains("StepRight"));
    assert!(dot.contains("N: 42"));
    assert!(dot.contains("Q: 0.750"));
    assert!(dot.contains("Root (Agent 0)"));
    assert!(dot.contains("Node #1 (Unexpanded)"));

    // Render to SVG and verify SVG output
    let svg = render_svg(&tree, root, &options).expect("render_svg failed");
    assert!(
        svg.contains("<svg"),
        "SVG output must contain <svg> root element"
    );
    assert!(
        svg.contains("StepRight"),
        "SVG output must contain action text"
    );

    // Render to PNG and verify PNG magic bytes
    let png = render_png(&tree, root, &options).expect("render_png failed");
    assert_eq!(
        &png[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );
}

#[test]
fn test_artificial_3_node_branching_tree() {
    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree: TreeStore<&'static str, [f32; 2], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    // Root expands 2 actions: "ActionA" (edge 0) and "ActionB" (edge 1)
    let _ = tree.expand_node(root, &["ActionA", "ActionB"]);
    let n1 = tree.insert_node(EdgeId(0), AgentId(1));
    let _n2 = tree.insert_node(EdgeId(1), AgentId(1));

    // Edge 0 (ActionA): heavily visited (PV path)
    tree.stats.visits[0] = 80;
    tree.stats.priors[0] = 0.75;
    tree.stats.mean_value[0] = [0.65, -0.65];

    // Edge 1 (ActionB): lower visits
    tree.stats.visits[1] = 20;
    tree.stats.priors[1] = 0.25;
    tree.stats.mean_value[1] = [-0.40, 0.40];

    // Node 1 expands 1 action: "ActionC" (edge 2) leading to Terminal Node 3
    let _ = tree.expand_node(n1, &["ActionC"]);
    let n3 = tree.insert_node(EdgeId(2), AgentId(0));
    tree.mark_terminal(n3);

    tree.stats.visits[2] = 70;
    tree.stats.priors[2] = 1.0;
    tree.stats.mean_value[2] = [1.0, -1.0];

    let options = TreeVisualizerOptions::new()
        .with_rankdir(RankDir::TopToBottom)
        .with_highlight_pv(true);

    let dot = to_dot(&tree, root, &options);
    println!("--- 3-Node Branching Tree DOT ---\n{dot}");

    let parsed = graphviz_rust::parse(&dot);
    assert!(
        parsed.is_ok(),
        "Generated DOT must be syntactically valid: {:?}",
        parsed.err()
    );

    // Verify PV highlighting: Edge 0 and Edge 2 should have penwidth "2.8" and PV color "#2563EB"
    assert!(
        dot.contains("#2563EB"),
        "PV path should be highlighted in blue"
    );
    assert!(
        dot.contains("Terminal"),
        "Node 3 should be marked as Terminal"
    );
    assert!(dot.contains("ActionA"));
    assert!(dot.contains("ActionB"));
    assert!(dot.contains("ActionC"));

    // Render to SVG
    let svg = render_svg(&tree, root, &options).expect("render_svg failed");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_artificial_ismcts_availability_tree() {
    let stats = IsmctsStats::<2>::new();
    let mut tree: TreeStore<&'static str, [f32; 2], IsmctsStats<2>, ()> =
        TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    let _ = tree.expand_node(root, &["PlayCard", "Pass"]);
    let _n1 = tree.insert_node(EdgeId(0), AgentId(1));
    let _n2 = tree.insert_node(EdgeId(1), AgentId(1));

    // PlayCard has high availability, Pass was rarely available
    tree.stats.visits[0] = 50;
    tree.stats.avail_visits[0] = 100;
    tree.stats.priors[0] = 0.8;
    tree.stats.mean_value[0] = [0.55, -0.55];

    tree.stats.visits[1] = 5;
    tree.stats.avail_visits[1] = 10;
    tree.stats.priors[1] = 0.2;
    tree.stats.mean_value[1] = [-0.80, 0.80];

    let options = TreeVisualizerOptions::new();
    let dot = to_dot(&tree, root, &options);

    println!("--- ISMCTS Tree DOT ---\n{dot}");

    assert!(dot.contains("avail: 100"), "Should show avail visits");
    assert!(dot.contains("avail: 10"), "Should show avail visits");

    let svg = render_svg(&tree, root, &options).expect("render_svg failed");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_artificial_stochastic_step_delta_tree() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<&'static str, [f32; 1], _, &'static str> =
        TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    // Edge 0: "RollDice"
    let _ = tree.expand_node(root, &["RollDice"]);

    // Stochastic branches for outcome "Roll_1" and "Roll_6"
    let (_n1, is_new1) = tree.get_or_insert_child(EdgeId(0), &"Roll_1", AgentId(1));
    let (_n2, is_new2) = tree.get_or_insert_child(EdgeId(0), &"Roll_6", AgentId(1));
    assert!(is_new1);
    assert!(is_new2);

    tree.stats.visits[0] = 30;
    tree.stats.mean_value[0] = [0.5];

    let options = TreeVisualizerOptions::new();
    let dot = to_dot(&tree, root, &options);

    println!("--- Stochastic Delta Tree DOT ---\n{dot}");

    assert!(dot.contains("Roll_1"), "Should display delta branch Roll_1");
    assert!(dot.contains("Roll_6"), "Should display delta branch Roll_6");

    let svg = render_svg(&tree, root, &options).expect("render_svg failed");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_depth_cutoff_and_min_visits_filtering() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<&'static str, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    let _ = tree.expand_node(root, &["MoveA", "MoveB"]);
    let n1 = tree.insert_node(EdgeId(0), AgentId(1));
    let _n2 = tree.insert_node(EdgeId(1), AgentId(1));

    tree.stats.visits[0] = 50;
    tree.stats.visits[1] = 2; // Below min_visits threshold

    let _ = tree.expand_node(n1, &["DeepMove"]);
    let _n3 = tree.insert_node(EdgeId(2), AgentId(0));
    tree.stats.visits[2] = 45;

    // 1. Min visits filter (threshold = 10): MoveB should be filtered out
    let options_filtered = TreeVisualizerOptions::new().with_min_visits(10);
    let dot_filtered = to_dot(&tree, root, &options_filtered);
    assert!(dot_filtered.contains("MoveA"));
    assert!(
        !dot_filtered.contains("MoveB"),
        "MoveB should be filtered by min_visits"
    );

    // 2. Max depth cutoff (depth = 1): DeepMove should not be rendered
    let options_depth = TreeVisualizerOptions::new().with_max_depth(1);
    let dot_depth = to_dot(&tree, root, &options_depth);
    assert!(dot_depth.contains("MoveA"));
    assert!(
        !dot_depth.contains("DeepMove"),
        "DeepMove exceeds max depth 1"
    );
}

#[test]
fn test_render_chance_nodes_option() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<&'static str, [f32; 1], _, &'static str> =
        TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    let _ = tree.expand_node(root, &["RollDice"]);
    let (_n1, _) = tree.get_or_insert_child(EdgeId(0), &"Heads", AgentId(1));
    let (_n2, _) = tree.get_or_insert_child(EdgeId(0), &"Tails", AgentId(1));

    tree.stats.visits[0] = 50;
    tree.stats.mean_value[0] = [0.5];

    let options = TreeVisualizerOptions::new().with_render_chance_nodes(true);
    let dot = to_dot(&tree, root, &options);

    println!("--- Chance Nodes Tree DOT ---\n{dot}");

    // Should contain virtual chance node c_0_0 with circle shape and orange color
    assert!(
        dot.contains("c_0_0"),
        "DOT should define virtual chance node c_0_0"
    );
    assert!(
        dot.contains("shape=\"circle\""),
        "Chance node must be shaped circle"
    );
    assert!(
        dot.contains("#FB923C"),
        "Chance node fillcolor must be orange"
    );
    assert!(
        dot.contains("n_0 -> c_0_0"),
        "Parent should link to chance node"
    );
    assert!(
        dot.contains("c_0_0 -> n_1"),
        "Chance node should link to first child"
    );
    assert!(
        dot.contains("c_0_0 -> n_2"),
        "Chance node should link to second child"
    );

    let svg = render_svg(&tree, root, &options).expect("render_svg failed");
    assert!(svg.contains("<svg"));
}
