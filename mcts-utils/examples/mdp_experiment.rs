//! Experiment running MCTS on GraphEnv MDPs (Chain-4 and Y-shaped) and rendering side-by-side.

use std::collections::HashMap;
use std::path::Path;

use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentId, Evaluation, GraphEnv, Model};
use mcts_utils::{
    RankDir, TreeVisualizerOptions, graph_env_to_dot, render_graph_env_svg, write_dot_file,
    write_png_file, write_svg_file,
};

/// Simple neutral model supplying uniform priors and zero heuristic values.
#[derive(Clone, Default)]
struct NeutralModel;

impl Model<u32> for NeutralModel {
    fn evaluate(&self, _s: &u32) -> Evaluation {
        // Uniform priors will be normalized by backup
        Evaluation {
            priors: vec![1.0, 1.0],
            values: vec![0.0],
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir =
        Path::new("/home/pankaj/.gemini/antigravity/brain/a0eaac1c-2837-489c-a5c1-8bf3e3b743b0");
    std::fs::create_dir_all(out_dir)?;

    println!("============================================================");
    println!("        MCTS EXPERIMENT: GRAPH-BASED MDPS (10 ITERATIONS)   ");
    println!("============================================================");

    // =========================================================================
    // 1. CHAIN OF 4 STATES (S0 -> S1 -> S2 -> S3 [Terminal +1.0])
    // =========================================================================
    println!("\n--- [1] 4-State Chain MDP ---");
    let mut chain_env = GraphEnv::<1>::new(0);
    chain_env.add_transition(0, 0, 1, [0.0], false);
    chain_env.add_transition(1, 0, 2, [0.0], false);
    chain_env.add_transition(2, 0, 3, [1.0], true);

    chain_env.set_actions(0, vec![0]);
    chain_env.set_actions(1, vec![0]);
    chain_env.set_actions(2, vec![0]);
    chain_env.set_actions(3, vec![]);
    chain_env.set_terminal(3, true);

    let mut chain_action_names = HashMap::new();
    chain_action_names.insert(0, "Advance".to_string());

    // Render Chain MDP Graph
    let chain_mdp_dot = graph_env_to_dot(&chain_env, "Chain_MDP", Some(&chain_action_names));
    let chain_mdp_svg = render_graph_env_svg(&chain_env, "Chain_MDP", Some(&chain_action_names))?;
    std::fs::write(out_dir.join("chain_mdp.dot"), &chain_mdp_dot)?;
    std::fs::write(out_dir.join("chain_mdp.svg"), &chain_mdp_svg)?;
    println!("Generated Chain MDP diagram (3 transitions, 1 terminal goal).");

    // Run MCTS for 10 iterations
    let model = NeutralModel;
    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.414 };
    let backup = VectorBackup::<1>::default();

    let mut chain_tree = TreeStore::with_capacity(20, 20, MultiAgentPuctStats::<1>::new());
    let chain_root = chain_tree.insert_root(AgentId(0));

    SequentialScheduler.search(
        &mut chain_tree,
        &chain_env,
        &model,
        &selection,
        &backup,
        chain_root,
        &0,
        10,
    );

    println!("MCTS completed 10 iterations on Chain MDP.");
    println!("  Total tree nodes allocated: {}", chain_tree.num_nodes());
    println!("  Total tree edges allocated: {}", chain_tree.num_edges());

    let chain_opts = TreeVisualizerOptions::new()
        .with_top_nodes(10)
        .with_highlight_pv(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_action_formatter(|&a| {
            if a == 0 {
                "Advance".to_string()
            } else {
                format!("a_{a}")
            }
        });

    write_svg_file(
        &chain_tree,
        chain_root,
        &chain_opts,
        out_dir.join("chain_tree_top10.svg"),
    )?;
    write_png_file(
        &chain_tree,
        chain_root,
        &chain_opts,
        out_dir.join("chain_tree_top10.png"),
    )?;
    write_dot_file(
        &chain_tree,
        chain_root,
        &chain_opts,
        out_dir.join("chain_tree_top10.dot"),
    )?;

    // =========================================================================
    // 2. Y-SHAPED MDP (S0 branches into Left -> S3 [+1.0] and Right -> S4 [+2.0])
    // =========================================================================
    println!("\n--- [2] Y-Shaped MDP ---");
    let mut y_env = GraphEnv::<1>::new(0);
    // S0 -> Left (0) -> S1, Right (1) -> S2
    y_env.add_transition(0, 0, 1, [0.0], false);
    y_env.add_transition(0, 1, 2, [0.0], false);

    // S1 -> FinishLeft (0) -> S3 (Terminal +1.0)
    y_env.add_transition(1, 0, 3, [1.0], true);

    // S2 -> FinishRight (1) -> S4 (Terminal +2.0)
    y_env.add_transition(2, 1, 4, [2.0], true);

    y_env.set_actions(0, vec![0, 1]);
    y_env.set_actions(1, vec![0]);
    y_env.set_actions(2, vec![1]);
    y_env.set_actions(3, vec![]);
    y_env.set_actions(4, vec![]);
    y_env.set_terminal(3, true);
    y_env.set_terminal(4, true);

    let mut y_action_names = HashMap::new();
    y_action_names.insert(0, "Left / FinishL".to_string());
    y_action_names.insert(1, "Right / FinishR".to_string());

    // Render Y-Shaped MDP Graph
    let y_mdp_dot = graph_env_to_dot(&y_env, "Y_Shaped_MDP", Some(&y_action_names));
    let y_mdp_svg = render_graph_env_svg(&y_env, "Y_Shaped_MDP", Some(&y_action_names))?;
    std::fs::write(out_dir.join("y_shaped_mdp.dot"), &y_mdp_dot)?;
    std::fs::write(out_dir.join("y_shaped_mdp.svg"), &y_mdp_svg)?;
    println!("Generated Y-Shaped MDP diagram (2 branches: Left=+1.0, Right=+2.0).");

    // Run MCTS for 10 iterations
    let mut y_tree = TreeStore::with_capacity(20, 20, MultiAgentPuctStats::<1>::new());
    let y_root = y_tree.insert_root(AgentId(0));

    SequentialScheduler.search(
        &mut y_tree,
        &y_env,
        &model,
        &selection,
        &backup,
        y_root,
        &0,
        10,
    );

    println!("MCTS completed 10 iterations on Y-Shaped MDP.");
    println!("  Total tree nodes allocated: {}", y_tree.num_nodes());
    println!("  Total tree edges allocated: {}", y_tree.num_edges());

    let y_opts = TreeVisualizerOptions::new()
        .with_top_nodes(10)
        .with_highlight_pv(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_action_formatter(|&a| match a {
            0 => "Left / FinishL".to_string(),
            1 => "Right / FinishR".to_string(),
            _ => format!("a_{a}"),
        });

    write_svg_file(
        &y_tree,
        y_root,
        &y_opts,
        out_dir.join("y_shaped_tree_top10.svg"),
    )?;
    write_png_file(
        &y_tree,
        y_root,
        &y_opts,
        out_dir.join("y_shaped_tree_top10.png"),
    )?;
    write_dot_file(
        &y_tree,
        y_root,
        &y_opts,
        out_dir.join("y_shaped_tree_top10.dot"),
    )?;

    // Also run for 25 iterations to show the crossover where Right (+2.0) is discovered!
    let mut y_tree_25 = TreeStore::with_capacity(20, 20, MultiAgentPuctStats::<1>::new());
    let y_root_25 = y_tree_25.insert_root(AgentId(0));
    SequentialScheduler.search(
        &mut y_tree_25,
        &y_env,
        &model,
        &selection,
        &backup,
        y_root_25,
        &0,
        25,
    );
    write_svg_file(
        &y_tree_25,
        y_root_25,
        &y_opts,
        out_dir.join("y_shaped_tree_25.svg"),
    )?;
    write_png_file(
        &y_tree_25,
        y_root_25,
        &y_opts,
        out_dir.join("y_shaped_tree_25.png"),
    )?;

    // =========================================================================
    // 3. Print Statistical Breakdown
    // =========================================================================
    println!("\n=== CHAIN MDP SEARCH STATISTICS ===");
    for e in 0..chain_tree.num_edges() {
        let edge = mcts_engine::tree_store::EdgeId(e as u32);
        let action = chain_tree.edge_action(edge);
        let visits = chain_tree.stats.visits[e];
        let q = chain_tree.stats.mean_value[e][0];
        let p = chain_tree.stats.priors[e];
        let child = chain_tree.edge_child(edge);
        println!(
            "  Edge {e}: action={action}, visits={visits}, Q={q:.3}, prior={p:.3}, child_node={:?}",
            child
        );
    }

    println!("\n=== Y-SHAPED MDP SEARCH STATISTICS (10 ITERATIONS) ===");
    for e in 0..y_tree.num_edges() {
        let edge = mcts_engine::tree_store::EdgeId(e as u32);
        let action = y_tree.edge_action(edge);
        let visits = y_tree.stats.visits[e];
        let q = y_tree.stats.mean_value[e][0];
        let p = y_tree.stats.priors[e];
        let child = y_tree.edge_child(edge);
        let action_name = if *action == 0 { "Left" } else { "Right" };
        println!(
            "  Edge {e} ({action_name}): visits={visits}, Q={q:.3}, prior={p:.3}, child_node={:?}",
            child
        );
    }

    println!("\n=== Y-SHAPED MDP SEARCH STATISTICS (25 ITERATIONS) ===");
    for e in 0..y_tree_25.num_edges() {
        let edge = mcts_engine::tree_store::EdgeId(e as u32);
        let action = y_tree_25.edge_action(edge);
        let visits = y_tree_25.stats.visits[e];
        let q = y_tree_25.stats.mean_value[e][0];
        let p = y_tree_25.stats.priors[e];
        let child = y_tree_25.edge_child(edge);
        let action_name = if *action == 0 { "Left" } else { "Right" };
        println!(
            "  Edge {e} ({action_name}): visits={visits}, Q={q:.3}, prior={p:.3}, child_node={:?}",
            child
        );
    }

    println!("\nAll diagrams, SVGs, PNGs, and DOT files saved to artifact directory.");
    Ok(())
}
