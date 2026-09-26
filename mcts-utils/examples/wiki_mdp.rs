//! Experiment running MCTS on the corrected Wikipedia Markov Decision Process (MDP) for 1000 iterations.
//!
//! State space: S = {S0, S1, S2}
//! Action space: A = {a0, a1} in all states
//! Transitions and Rewards:
//!   - (S0, a0) -> 0.5 S0 (R=0), 0.5 S2 (R=0)
//!   - (S0, a1) -> 1.0 S2 (R=0)
//!   - (S1, a0) -> 0.7 S0 (R=+5), 0.1 S1 (R=0), 0.2 S2 (R=0)  [E[R] = +3.5]
//!   - (S1, a1) -> 0.95 S1 (R=0), 0.05 S2 (R=0)               [E[R] = 0.0]
//!   - (S2, a0) -> 0.4 S0 (R=0), 0.6 S2 (R=0)                 [E[R] = 0.0]
//!   - (S2, a1) -> 0.3 S0 (R=-1), 0.3 S1 (R=0), 0.4 S2 (R=0) [E[R] = -0.3]

use std::cell::RefCell;
use std::fs;
use std::path::Path;

use graphviz_rust::cmd::{CommandArg, Format};
use graphviz_rust::exec;
use graphviz_rust::parse;
use graphviz_rust::printer::PrinterContext;

use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::{EdgeId, NodeId, TreeStore};
use mcts_traits::{AgentDynamics, AgentId, Evaluation, Model, StepOutcome};
use mcts_utils::{RankDir, TreeVisualizerOptions, render_svg, write_dot_file};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Formal 3-state, 2-action Markov Decision Process dynamics.
#[derive(Debug, Clone)]
pub struct WikiMdp {
    rng: RefCell<StdRng>,
}

impl WikiMdp {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl AgentDynamics for WikiMdp {
    type State = u32; // 0: S0, 1: S1, 2: S2
    type Action = u32; // 0: a0, 1: a1
    type Reward = [f32; 1];
    type StepDelta = u32; // 0: S0, 1: S1, 2: S2

    fn initial(&self) -> Self::State {
        0
    }

    fn actions(&self, _s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        out.extend_from_slice(&[0, 1]);
    }

    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        let mut rng = self.rng.borrow_mut();
        let p: f64 = rng.r#gen();

        let (next_state, reward) = match (*s, *action) {
            // (S0, a0) -> 50% S0 (R=0), 50% S2 (R=0)
            (0, 0) => {
                if p < 0.5 {
                    (0, 0.0)
                } else {
                    (2, 0.0)
                }
            }
            // (S0, a1) -> 100% S2 (R=0)
            (0, 1) => (2, 0.0),

            // (S1, a0) -> 70% S0 (R=+5), 10% S1 (R=0), 20% S2 (R=0)
            (1, 0) => {
                if p < 0.7 {
                    (0, 5.0)
                } else if p < 0.8 {
                    (1, 0.0)
                } else {
                    (2, 0.0)
                }
            }

            // (S1, a1) -> 95% S1 (R=0), 5% S2 (R=0)
            (1, 1) => {
                if p < 0.95 {
                    (1, 0.0)
                } else {
                    (2, 0.0)
                }
            }

            // (S2, a0) -> 40% S0 (R=0), 60% S2 (R=0)
            (2, 0) => {
                if p < 0.4 {
                    (0, 0.0)
                } else {
                    (2, 0.0)
                }
            }

            // (S2, a1) -> 30% S0 (R=-1), 30% S1 (R=0), 40% S2 (R=0)
            (2, 1) => {
                if p < 0.3 {
                    (0, -1.0)
                } else if p < 0.6 {
                    (1, 0.0)
                } else {
                    (2, 0.0)
                }
            }

            _ => panic!("WikiMdp: illegal state-action pair ({}, {})", *s, *action),
        };

        *s = next_state;
        StepOutcome::with_delta([reward], next_state, false)
    }

    fn current_agent(&self, _s: &Self::State) -> AgentId {
        AgentId(0)
    }
}

/// Neutral model providing uniform action priors (0.5, 0.5) and zero heuristic values.
#[derive(Clone, Default)]
struct NeutralModel;

impl Model<u32> for NeutralModel {
    fn evaluate(&self, _s: &u32) -> Evaluation {
        Evaluation {
            priors: vec![0.5, 0.5],
            values: vec![0.0],
        }
    }
}

/// Generates the DOT specification for the corrected 3-state, 2-action MDP diagram.
fn generate_wiki_mdp_dot() -> String {
    r##"digraph WikiMDP {
    rankdir=LR;
    bgcolor="transparent";
    nodesep="0.6";
    ranksep="0.8";
    splines="spline";
    node [fontname="Helvetica,Arial,sans-serif"];
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];

    // States (Green circles)
    s0 [label=<<B>S₀</B><BR/><FONT POINT-SIZE="9" COLOR="#14532D">Start State</FONT>>, shape=circle, style="filled", fillcolor="#86EFAC", color="#16A34A", penwidth=2.5, width=0.9, fixedsize=true, fontcolor="#14532D"];
    s1 [label=<<B>S₁</B><BR/><FONT POINT-SIZE="9" COLOR="#14532D">High Reward</FONT>>, shape=circle, style="filled", fillcolor="#86EFAC", color="#16A34A", penwidth=2.5, width=0.9, fixedsize=true, fontcolor="#14532D"];
    s2 [label=<<B>S₂</B><BR/><FONT POINT-SIZE="9" COLOR="#14532D">Gateway</FONT>>, shape=circle, style="filled", fillcolor="#86EFAC", color="#16A34A", penwidth=2.5, width=0.9, fixedsize=true, fontcolor="#14532D"];

    // Action nodes (Orange circles matching Wikipedia)
    a_s0_0 [label="a₀", shape=circle, style="filled", fillcolor="#FB923C", color="#EA580C", penwidth=2.0, width=0.45, fixedsize=true, fontsize=11, fontcolor="#FFFFFF"];
    a_s0_1 [label="a₁", shape=circle, style="filled", fillcolor="#FB923C", color="#EA580C", penwidth=2.0, width=0.45, fixedsize=true, fontsize=11, fontcolor="#FFFFFF"];
    a_s1_0 [label="a₀", shape=circle, style="filled", fillcolor="#FB923C", color="#EA580C", penwidth=2.0, width=0.45, fixedsize=true, fontsize=11, fontcolor="#FFFFFF"];
    a_s1_1 [label="a₁", shape=circle, style="filled", fillcolor="#FB923C", color="#EA580C", penwidth=2.0, width=0.45, fixedsize=true, fontsize=11, fontcolor="#FFFFFF"];
    a_s2_0 [label="a₀", shape=circle, style="filled", fillcolor="#FB923C", color="#EA580C", penwidth=2.0, width=0.45, fixedsize=true, fontsize=11, fontcolor="#FFFFFF"];
    a_s2_1 [label="a₁", shape=circle, style="filled", fillcolor="#FB923C", color="#EA580C", penwidth=2.0, width=0.45, fixedsize=true, fontsize=11, fontcolor="#FFFFFF"];

    // Edges from states to action nodes
    s0 -> a_s0_0 [penwidth=1.6, color="#64748B", arrowhead="vee"];
    s0 -> a_s0_1 [penwidth=2.4, color="#2563EB", arrowhead="vee"];
    s1 -> a_s1_0 [penwidth=2.4, color="#16A34A", arrowhead="vee"];
    s1 -> a_s1_1 [penwidth=1.6, color="#64748B", arrowhead="vee"];
    s2 -> a_s2_0 [penwidth=1.6, color="#64748B", arrowhead="vee"];
    s2 -> a_s2_1 [penwidth=2.4, color="#2563EB", arrowhead="vee"];

    // S0 Action transitions
    a_s0_0 -> s0 [label="P = 0.5\nR = 0", color="#64748B", fontcolor="#334155", penwidth=1.4, arrowhead="vee"];
    a_s0_0 -> s2 [label="P = 0.5\nR = 0", color="#64748B", fontcolor="#334155", penwidth=1.4, arrowhead="vee"];
    a_s0_1 -> s2 [label="P = 1.0\nR = 0", color="#2563EB", fontcolor="#1D4ED8", penwidth=2.4, arrowhead="vee"];

    // S1 Action transitions
    a_s1_0 -> s0 [label="P = 0.7\nR = +5", color="#16A34A", fontcolor="#15803D", fontname="Helvetica-Bold", penwidth=2.6, arrowhead="vee"];
    a_s1_0 -> s1 [label="P = 0.1\nR = 0", color="#94A3B8", fontcolor="#64748B", penwidth=1.2, arrowhead="vee"];
    a_s1_0 -> s2 [label="P = 0.2\nR = 0", color="#94A3B8", fontcolor="#64748B", penwidth=1.2, arrowhead="vee"];
    a_s1_1 -> s1 [label="P = 0.95\nR = 0", color="#64748B", fontcolor="#334155", penwidth=1.6, arrowhead="vee"];
    a_s1_1 -> s2 [label="P = 0.05\nR = 0", color="#CBD5E1", fontcolor="#94A3B8", penwidth=1.0, arrowhead="vee"];

    // S2 Action transitions
    a_s2_0 -> s0 [label="P = 0.4\nR = 0", color="#64748B", fontcolor="#334155", penwidth=1.4, arrowhead="vee"];
    a_s2_0 -> s2 [label="P = 0.6\nR = 0", color="#64748B", fontcolor="#334155", penwidth=1.6, arrowhead="vee"];
    a_s2_1 -> s0 [label="P = 0.3\nR = -1", color="#DC2626", fontcolor="#B91C1C", fontname="Helvetica-Bold", penwidth=1.8, arrowhead="vee"];
    a_s2_1 -> s1 [label="P = 0.3\nR = 0", color="#2563EB", fontcolor="#1D4ED8", fontname="Helvetica-Bold", penwidth=2.4, arrowhead="vee"];
    a_s2_1 -> s2 [label="P = 0.4\nR = 0", color="#64748B", fontcolor="#334155", penwidth=1.4, arrowhead="vee"];
}"##.to_string()
}

fn run_mcts_experiment(
    c_puct: f32,
    gamma: f32,
    num_iterations: usize,
    seed: u64,
) -> (
    TreeStore<u32, [f32; 1], MultiAgentPuctStats<1>, u32>,
    NodeId,
) {
    let dynamics = WikiMdp::new(seed);
    let model = NeutralModel;
    let selection = MultiAgentPuctSelection::<1> { c_puct };
    let backup = VectorBackup::<1>::new(gamma);

    let mut tree = TreeStore::with_capacity(8192, 8192, MultiAgentPuctStats::<1>::new());
    let root = tree.insert_root(AgentId(0));

    SequentialScheduler.search(
        &mut tree,
        &dynamics,
        &model,
        &selection,
        &backup,
        root,
        &0,
        num_iterations,
    );

    (tree, root)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir =
        Path::new("/home/pankaj/.gemini/antigravity/brain/a0eaac1c-2837-489c-a5c1-8bf3e3b743b0");
    fs::create_dir_all(out_dir)?;

    println!("============================================================");
    println!("      WIKIPEDIA MDP: MCTS 1000 ITERATIONS EXPERIMENT        ");
    println!("============================================================");

    // 1. Render Wikipedia MDP State Graph
    let wiki_dot = generate_wiki_mdp_dot();
    fs::write(out_dir.join("wiki_mdp.dot"), &wiki_dot)?;

    let parsed_mdp_graph =
        parse(&wiki_dot).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let wiki_mdp_svg_bytes = exec(
        parsed_mdp_graph,
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )?;
    let wiki_mdp_svg = String::from_utf8(wiki_mdp_svg_bytes)?;
    fs::write(out_dir.join("wiki_mdp.svg"), &wiki_mdp_svg)?;
    println!("1. Generated and rendered Wikipedia MDP diagram.");

    // 2. Run Baseline MCTS with c_puct = 1.414 (1000 iterations, gamma = 0.95)
    println!("\n2. Running Baseline MCTS (c_puct = 1.414, gamma = 0.95, 1000 iters)...");
    let (tree_c1, root_c1) = run_mcts_experiment(1.414, 0.95, 1000, 42);
    let n_a0_c1 = tree_c1.stats.visits[EdgeId(0).as_usize()];
    let q_a0_c1 = tree_c1.stats.mean_value[EdgeId(0).as_usize()][0];
    let n_a1_c1 = tree_c1.stats.visits[EdgeId(1).as_usize()];
    let q_a1_c1 = tree_c1.stats.mean_value[EdgeId(1).as_usize()][0];
    println!("   Baseline Root Edge Statistics:");
    println!("   - a0: visits = {n_a0_c1:4}, Q = {q_a0_c1:+.4}");
    println!("   - a1: visits = {n_a1_c1:4}, Q = {q_a1_c1:+.4}");

    // Render Top 10 for Baseline with Chance Nodes
    let top10_opts = TreeVisualizerOptions::new()
        .with_top_nodes(10)
        .with_highlight_pv(true)
        .with_render_chance_nodes(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_graph_id("WikiMDP_MCTS_Top10_c1".to_string())
        .with_action_formatter(|&a| match a {
            0 => "a₀".to_string(),
            1 => "a₁".to_string(),
            _ => format!("a_{a}"),
        })
        .with_delta_formatter(|&d| match d {
            0 => "S₀".to_string(),
            1 => "S₁".to_string(),
            2 => "S₂".to_string(),
            _ => format!("S_{d}"),
        })
        .with_custom_node_label(|node, status, _agent| {
            if node == NodeId(0) {
                Some("Root (S₀)".to_string())
            } else {
                Some(format!("Node {} ({:?})", node.0, status))
            }
        });
    let top10_svg = render_svg(&tree_c1, root_c1, &top10_opts)?;
    fs::write(out_dir.join("wiki_tree_top10.svg"), &top10_svg)?;
    write_dot_file(
        &tree_c1,
        root_c1,
        &top10_opts,
        out_dir.join("wiki_tree_top10.dot"),
    )?;

    // Render Top 25 for Baseline with Chance Nodes
    let top25_opts = TreeVisualizerOptions::new()
        .with_top_nodes(25)
        .with_highlight_pv(true)
        .with_render_chance_nodes(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_graph_id("WikiMDP_MCTS_Top25_c1".to_string())
        .with_action_formatter(|&a| match a {
            0 => "a₀".to_string(),
            1 => "a₁".to_string(),
            _ => format!("a_{a}"),
        })
        .with_delta_formatter(|&d| match d {
            0 => "S₀".to_string(),
            1 => "S₁".to_string(),
            2 => "S₂".to_string(),
            _ => format!("S_{d}"),
        });
    let top25_svg = render_svg(&tree_c1, root_c1, &top25_opts)?;
    fs::write(out_dir.join("wiki_tree_top25.svg"), &top25_svg)?;
    write_dot_file(
        &tree_c1,
        root_c1,
        &top25_opts,
        out_dir.join("wiki_tree_top25.dot"),
    )?;

    // 3. Run Scaled Exploration MCTS with c_puct = 10.0 (1000 iterations, gamma = 0.95)
    println!("\n3. Running Scaled MCTS (c_puct = 10.0, gamma = 0.95, 1000 iters)...");
    let (tree_c10, root_c10) = run_mcts_experiment(10.0, 0.95, 1000, 42);
    let n_a0_c10 = tree_c10.stats.visits[EdgeId(0).as_usize()];
    let q_a0_c10 = tree_c10.stats.mean_value[EdgeId(0).as_usize()][0];
    let n_a1_c10 = tree_c10.stats.visits[EdgeId(1).as_usize()];
    let q_a1_c10 = tree_c10.stats.mean_value[EdgeId(1).as_usize()][0];
    println!("   Scaled Root Edge Statistics:");
    println!("   - a0: visits = {n_a0_c10:4}, Q = {q_a0_c10:+.4}");
    println!("   - a1: visits = {n_a1_c10:4}, Q = {q_a1_c10:+.4}");

    let top10_c10_opts = TreeVisualizerOptions::new()
        .with_top_nodes(10)
        .with_highlight_pv(true)
        .with_render_chance_nodes(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_graph_id("WikiMDP_MCTS_Top10_c10".to_string())
        .with_action_formatter(|&a| match a {
            0 => "a₀".to_string(),
            1 => "a₁".to_string(),
            _ => format!("a_{a}"),
        })
        .with_delta_formatter(|&d| match d {
            0 => "S₀".to_string(),
            1 => "S₁".to_string(),
            2 => "S₂".to_string(),
            _ => format!("S_{d}"),
        })
        .with_custom_node_label(|node, status, _agent| {
            if node == NodeId(0) {
                Some("Root (S₀)".to_string())
            } else {
                Some(format!("Node {} ({:?})", node.0, status))
            }
        });
    let top10_c10_svg = render_svg(&tree_c10, root_c10, &top10_c10_opts)?;
    fs::write(out_dir.join("wiki_tree_c10_top10.svg"), &top10_c10_svg)?;
    write_dot_file(
        &tree_c10,
        root_c10,
        &top10_c10_opts,
        out_dir.join("wiki_tree_c10_top10.dot"),
    )?;

    println!("\nVisualization artifacts exported successfully.");
    Ok(())
}
