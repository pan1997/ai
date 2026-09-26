//! Example demonstrating MCTS tree visualization and file rendering.

use std::path::Path;

use mcts_engine::selection::MultiAgentPuctStats;
use mcts_engine::selection::ismcts::IsmctsStats;
use mcts_engine::tree_store::{EdgeId, TreeStore};
use mcts_traits::AgentId;
use mcts_utils::{RankDir, TreeVisualizerOptions, write_dot_file, write_png_file, write_svg_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir =
        Path::new("/home/pankaj/.gemini/antigravity/brain/a0eaac1c-2837-489c-a5c1-8bf3e3b743b0");
    std::fs::create_dir_all(out_dir)?;

    println!("Rendering visual tree artifacts to: {}", out_dir.display());

    // 1. Two-Node Tree
    {
        let stats = MultiAgentPuctStats::<1>::new();
        let mut tree: TreeStore<&'static str, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
        let root = tree.insert_root(AgentId(0));
        let e0 = tree.expand_node(root, &["StepRight"]);
        let _ = tree.insert_node(e0, AgentId(1));
        tree.stats.visits[0] = 42;
        tree.stats.priors[0] = 1.0;
        tree.stats.mean_value[0] = [0.75];

        let options = TreeVisualizerOptions::new().with_rankdir(RankDir::TopToBottom);
        write_svg_file(&tree, root, &options, out_dir.join("sample_2_node.svg"))?;
        write_png_file(&tree, root, &options, out_dir.join("sample_2_node.png"))?;
        write_dot_file(&tree, root, &options, out_dir.join("sample_2_node.dot"))?;
    }

    // 2. Three-Node Branching Tree with Principal Variation (PV) Highlighting
    {
        let stats = MultiAgentPuctStats::<2>::new();
        let mut tree: TreeStore<&'static str, [f32; 2], _> =
            TreeStore::with_capacity(10, 10, stats);
        let root = tree.insert_root(AgentId(0));

        let _ = tree.expand_node(root, &["ActionA", "ActionB"]);
        let n1 = tree.insert_node(EdgeId(0), AgentId(1));
        let _n2 = tree.insert_node(EdgeId(1), AgentId(1));

        tree.stats.visits[0] = 80;
        tree.stats.priors[0] = 0.75;
        tree.stats.mean_value[0] = [0.65, -0.65];

        tree.stats.visits[1] = 20;
        tree.stats.priors[1] = 0.25;
        tree.stats.mean_value[1] = [-0.40, 0.40];

        let _ = tree.expand_node(n1, &["ActionC"]);
        let n3 = tree.insert_node(EdgeId(2), AgentId(0));
        tree.mark_terminal(n3);

        tree.stats.visits[2] = 70;
        tree.stats.priors[2] = 1.0;
        tree.stats.mean_value[2] = [1.0, -1.0];

        let options = TreeVisualizerOptions::new()
            .with_rankdir(RankDir::TopToBottom)
            .with_highlight_pv(true);

        write_svg_file(
            &tree,
            root,
            &options,
            out_dir.join("sample_branching_pv.svg"),
        )?;
        write_png_file(
            &tree,
            root,
            &options,
            out_dir.join("sample_branching_pv.png"),
        )?;
        write_dot_file(
            &tree,
            root,
            &options,
            out_dir.join("sample_branching_pv.dot"),
        )?;
    }

    // 3. ISMCTS Availability Tree
    {
        let stats = IsmctsStats::<2>::new();
        let mut tree: TreeStore<&'static str, [f32; 2], IsmctsStats<2>, ()> =
            TreeStore::with_capacity(10, 10, stats);
        let root = tree.insert_root(AgentId(0));

        let _ = tree.expand_node(root, &["PlayCard", "Pass"]);
        let _n1 = tree.insert_node(EdgeId(0), AgentId(1));
        let _n2 = tree.insert_node(EdgeId(1), AgentId(1));

        tree.stats.visits[0] = 50;
        tree.stats.avail_visits[0] = 100;
        tree.stats.priors[0] = 0.8;
        tree.stats.mean_value[0] = [0.55, -0.55];

        tree.stats.visits[1] = 5;
        tree.stats.avail_visits[1] = 10;
        tree.stats.priors[1] = 0.2;
        tree.stats.mean_value[1] = [-0.80, 0.80];

        let options = TreeVisualizerOptions::new();
        write_svg_file(&tree, root, &options, out_dir.join("sample_ismcts.svg"))?;
        write_png_file(&tree, root, &options, out_dir.join("sample_ismcts.png"))?;
        write_dot_file(&tree, root, &options, out_dir.join("sample_ismcts.dot"))?;
    }

    // 4. Stochastic Chance Node (StepDelta Branches)
    {
        let stats = MultiAgentPuctStats::<1>::new();
        let mut tree: TreeStore<&'static str, [f32; 1], _, &'static str> =
            TreeStore::with_capacity(10, 10, stats);
        let root = tree.insert_root(AgentId(0));

        let _ = tree.expand_node(root, &["RollDice"]);
        let _ = tree.get_or_insert_child(EdgeId(0), &"Roll_1", AgentId(1));
        let _ = tree.get_or_insert_child(EdgeId(0), &"Roll_6", AgentId(1));

        tree.stats.visits[0] = 30;
        tree.stats.mean_value[0] = [0.5];

        let options = TreeVisualizerOptions::new();
        write_svg_file(
            &tree,
            root,
            &options,
            out_dir.join("sample_stochastic_delta.svg"),
        )?;
        write_png_file(
            &tree,
            root,
            &options,
            out_dir.join("sample_stochastic_delta.png"),
        )?;
        write_dot_file(
            &tree,
            root,
            &options,
            out_dir.join("sample_stochastic_delta.dot"),
        )?;
    }

    // 5. Node Filtering (TopNodes vs MinVisits)
    {
        let stats = MultiAgentPuctStats::<1>::new();
        let mut tree: TreeStore<&'static str, [f32; 1], _> =
            TreeStore::with_capacity(10, 10, stats);
        let root = tree.insert_root(AgentId(0));

        let _ = tree.expand_node(root, &["MoveA", "MoveB", "MoveC"]);
        let n1 = tree.insert_node(EdgeId(0), AgentId(1));
        let _n2 = tree.insert_node(EdgeId(1), AgentId(1));
        let _n3 = tree.insert_node(EdgeId(2), AgentId(1));

        tree.stats.visits[0] = 100;
        tree.stats.visits[1] = 15;
        tree.stats.visits[2] = 2; // Low visit branch

        let _ = tree.expand_node(n1, &["DeepMove1", "DeepMove2"]);
        let _n4 = tree.insert_node(EdgeId(3), AgentId(0));
        let _n5 = tree.insert_node(EdgeId(4), AgentId(0));
        tree.stats.visits[3] = 85;
        tree.stats.visits[4] = 15;

        // Render only Top 3 visited nodes (root is always included)
        let options_top = TreeVisualizerOptions::new().with_top_nodes(3);
        write_svg_file(
            &tree,
            root,
            &options_top,
            out_dir.join("sample_filtered_top3.svg"),
        )?;
        write_png_file(
            &tree,
            root,
            &options_top,
            out_dir.join("sample_filtered_top3.png"),
        )?;
    }

    println!("All visual samples successfully rendered and written!");
    Ok(())
}
