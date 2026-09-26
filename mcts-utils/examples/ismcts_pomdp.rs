//! Validation experiment for Single-Tree ISMCTS on a Y-shaped POMDP.
//!
//! # Problem Setup
//! - Two underlying MDPs: **MDP 0** and **MDP 1**, each forming a Y-maze.
//! - The agent has a 50% initial belief probability of being in either MDP.
//! - **Test 1 (Blind / Shared Tree)**:
//!   - Observations along the stem (`Advance`) are identical in both MDPs (`Uniform`).
//!   - At the split, `Left` yields +1.0 in MDP 0 (0.0 in MDP 1); `Right` yields +1.0 in MDP 1 (0.0 in MDP 0).
//!   - The single-tree ISMCTS information set merges both MDPs.
//!   - Expected return at the split: E[Q(Left)] = E[Q(Right)] = 0.50.
//!   - Neither move dominates.
//! - **Test 2 (Observable Cue / Information Set Splitting)**:
//!   - Just before the split, the agent receives an informative observation:
//!     $O_0$ in MDP 0, and $O_1$ in MDP 1.
//!   - Single-Tree ISMCTS branches at that node into two subtrees conditioned on $O_0$ and $O_1$.
//!   - In Subtree $O_0$, ISMCTS correctly finds `Left` as the dominant best move ($Q = 1.0$).
//!   - In Subtree $O_1$, ISMCTS correctly finds `Right` as the dominant best move ($Q = 1.0$).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::IsmctsScheduler;
use mcts_engine::selection::ismcts::{IsmctsSelection, IsmctsStats};
use mcts_engine::tree_store::{EdgeId, TreeStore};
use mcts_traits::belief::BeliefSampler;
use mcts_traits::{ActionModel, AgentDynamics, AgentId, Evaluation, GraphEnv, Model, StepOutcome};
use mcts_utils::{
    RankDir, TreeVisualizerOptions, graph_env_to_dot, render_graph_env_svg, write_dot_file,
    write_png_file, write_svg_file,
};

// =============================================================================
// POMDP DOMAIN DEFINITIONS
// =============================================================================

/// Actions available in the Y-maze POMDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    /// Move along corridor / stem towards the split decision junction.
    Advance = 0,
    /// Choose the left branch at the split.
    Left = 1,
    /// Choose the right branch at the split.
    Right = 2,
}

/// Navigation stage within the Y-maze.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Initial corridor position ($S_0$).
    Stem,
    /// Decision junction / split point ($S_1$).
    Split,
    /// Goal state ($S_{\text{term}}$).
    Terminal,
}

/// Underlying state of the POMDP: includes true hidden MDP identity and spatial position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PomdpState {
    /// Hidden MDP identity (0 or 1).
    pub mdp_id: u8,
    /// Current spatial step.
    pub step: Step,
}

/// Observation received upon state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Observation {
    /// Uninformative / identical observation across both MDPs (Test 1).
    #[default]
    Uniform,
    /// Informative cue indicating MDP 0 (Test 2).
    Obs0,
    /// Informative cue indicating MDP 1 (Test 2).
    Obs1,
}

/// Observation regime for the experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsMode {
    /// Test 1: Observations are identical across MDP 0 and MDP 1.
    Identical,
    /// Test 2: Observations distinguish MDP 0 ($O_0$) from MDP 1 ($O_1$).
    Distinct,
}

/// Agent transition dynamics for the Y-shaped POMDP.
pub struct PomdpDynamics {
    pub obs_mode: ObsMode,
}

impl AgentDynamics for PomdpDynamics {
    type State = PomdpState;
    type Action = Action;
    type Reward = [f32; 1];
    type StepDelta = Observation;

    fn initial(&self) -> Self::State {
        PomdpState {
            mdp_id: 0,
            step: Step::Stem,
        }
    }

    fn actions(&self, state: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        match state.step {
            Step::Stem => out.push(Action::Advance),
            Step::Split => {
                out.push(Action::Left);
                out.push(Action::Right);
            }
            Step::Terminal => {}
        }
    }

    fn expand_actions(&self, state: &Self::State, out: &mut Vec<Self::Action>) {
        self.actions(state, out);
    }

    fn current_agent(&self, _state: &Self::State) -> AgentId {
        AgentId(0)
    }

    fn step(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        match state.step {
            Step::Stem => {
                assert_eq!(*action, Action::Advance, "Stem only allows Advance");
                state.step = Step::Split;
                let delta = match self.obs_mode {
                    ObsMode::Identical => Observation::Uniform,
                    ObsMode::Distinct => {
                        if state.mdp_id == 0 {
                            Observation::Obs0
                        } else {
                            Observation::Obs1
                        }
                    }
                };
                StepOutcome::with_delta([0.0], delta, false)
            }
            Step::Split => {
                state.step = Step::Terminal;
                let reward = match *action {
                    Action::Left => {
                        if state.mdp_id == 0 {
                            [1.0]
                        } else {
                            [0.0]
                        }
                    }
                    Action::Right => {
                        if state.mdp_id == 0 {
                            [0.0]
                        } else {
                            [1.0]
                        }
                    }
                    Action::Advance => panic!("Advance invalid at Split"),
                };
                StepOutcome::with_delta(reward, Observation::Uniform, true)
            }
            Step::Terminal => panic!("Cannot step terminal state"),
        }
    }
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Independent belief sampler sampling MDP 0 and MDP 1 with 50% probability via PRNG.
pub struct RandomSampler {
    rng: StdRng,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl BeliefSampler for RandomSampler {
    type State = PomdpState;
    type Context = ();

    fn sample(&mut self, _context: &Self::Context) -> Self::State {
        let mdp_id = if self.rng.gen_bool(0.5) { 0 } else { 1 };
        PomdpState {
            mdp_id,
            step: Step::Stem,
        }
    }
}

/// Neutral model providing uniform action priors and 0.0 heuristic baseline value.
#[derive(Clone, Default)]
pub struct NeutralModel;

impl Model<PomdpState> for NeutralModel {
    fn evaluate(&self, _s: &PomdpState) -> Evaluation {
        Evaluation {
            priors: vec![0.5, 0.5],
            values: vec![0.0],
        }
    }
}

impl ActionModel<PomdpState, Action> for NeutralModel {
    fn evaluate_actions(&self, state: &PomdpState, actions: &[Action]) -> Evaluation {
        let n = actions.len();
        let priors = if n > 0 {
            vec![1.0 / n as f32; n]
        } else {
            Vec::new()
        };
        let mut eval = self.evaluate(state);
        eval.priors = priors;
        eval
    }
}

fn format_action(a: &Action) -> String {
    match a {
        Action::Advance => "Advance".to_string(),
        Action::Left => "Left".to_string(),
        Action::Right => "Right".to_string(),
    }
}

fn format_obs(obs: &Observation) -> String {
    match obs {
        Observation::Uniform => "Same Obs".to_string(),
        Observation::Obs0 => "O_0 (MDP 0)".to_string(),
        Observation::Obs1 => "O_1 (MDP 1)".to_string(),
    }
}

// =============================================================================
// MAIN EXPERIMENT RUNNER
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir =
        Path::new("/home/pankaj/.gemini/antigravity/brain/a0eaac1c-2837-489c-a5c1-8bf3e3b743b0");
    fs::create_dir_all(out_dir)?;

    println!("============================================================");
    println!("        SINGLE-TREE ISMCTS VALIDATION: Y-SHAPED POMDP       ");
    println!("============================================================");

    // -------------------------------------------------------------------------
    // 0. GENERATE UNDERLYING MDP & POMDP STRUCTURE DIAGRAMS
    // -------------------------------------------------------------------------
    println!("\n>>> [0] Generating Reference MDP & POMDP Diagrams...");

    let mut action_names = HashMap::new();
    action_names.insert(0, "Advance".to_string());
    action_names.insert(1, "Left".to_string());
    action_names.insert(2, "Right".to_string());

    // MDP 0
    let mut mdp0 = GraphEnv::<1>::new(0);
    mdp0.add_transition(0, 0, 1, [0.0], false);
    mdp0.add_transition(1, 1, 2, [1.0], true);
    mdp0.add_transition(1, 2, 3, [0.0], true);
    mdp0.set_actions(0, vec![0]);
    mdp0.set_actions(1, vec![1, 2]);
    mdp0.set_actions(2, vec![]);
    mdp0.set_actions(3, vec![]);
    mdp0.set_terminal(2, true);
    mdp0.set_terminal(3, true);

    let mdp0_svg = render_graph_env_svg(&mdp0, "MDP_0", Some(&action_names))?;
    fs::write(out_dir.join("pomdp_mdp0.svg"), &mdp0_svg)?;
    fs::write(
        out_dir.join("pomdp_mdp0.dot"),
        graph_env_to_dot(&mdp0, "MDP_0", Some(&action_names)),
    )?;

    // MDP 1
    let mut mdp1 = GraphEnv::<1>::new(0);
    mdp1.add_transition(0, 0, 1, [0.0], false);
    mdp1.add_transition(1, 1, 2, [0.0], true);
    mdp1.add_transition(1, 2, 3, [1.0], true);
    mdp1.set_actions(0, vec![0]);
    mdp1.set_actions(1, vec![1, 2]);
    mdp1.set_actions(2, vec![]);
    mdp1.set_actions(3, vec![]);
    mdp1.set_terminal(2, true);
    mdp1.set_terminal(3, true);

    let mdp1_svg = render_graph_env_svg(&mdp1, "MDP_1", Some(&action_names))?;
    fs::write(out_dir.join("pomdp_mdp1.svg"), &mdp1_svg)?;
    fs::write(
        out_dir.join("pomdp_mdp1.dot"),
        graph_env_to_dot(&mdp1, "MDP_1", Some(&action_names)),
    )?;

    // Unified POMDP Diagram
    let pomdp_dot = r##"digraph POMDP_Structure {
  rankdir=TB;
  bgcolor="#ffffff";
  nodesep=0.5;
  ranksep=0.6;
  splines=spline;
  node [fontname="Helvetica,Arial,sans-serif", fontsize=11];
  edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];

  c_init [shape=circle, style=filled, fillcolor="#FB923C", color="#EA580C", penwidth=1.5, width=0.25, height=0.25, label="", tooltip="Chance Node: 50% Belief"];
  
  subgraph cluster_mdp0 {
    label="MDP 0 (P = 0.5)";
    style="rounded,filled";
    fillcolor="#EFF6FF";
    color="#3B82F6";
    penwidth=1.5;

    s0_0 [shape=box, style="rounded,filled", fillcolor="#DBEAFE", color="#1D4ED8", label="S0 (Stem)\n[MDP 0]"];
    s1_0 [shape=box, style="rounded,filled", fillcolor="#DBEAFE", color="#1D4ED8", label="S1 (Split)\n[MDP 0]"];
    s2_0 [shape=box, style="rounded,filled", fillcolor="#DCFCE7", color="#059669", penwidth=2, label="S2 (Goal L)\nTerminal"];
    s3_0 [shape=box, style="rounded,filled", fillcolor="#FEE2E2", color="#DC2626", label="S3 (Goal R)\nTerminal"];

    s0_0 -> s1_0 [label="Advance\n(Obs: O_0)", color="#2563EB", penwidth=1.5];
    s1_0 -> s2_0 [label="Left\nR: +1.0", color="#059669", fontcolor="#059669", penwidth=2];
    s1_0 -> s3_0 [label="Right\nR: 0.0", color="#94A3B8"];
  }

  subgraph cluster_mdp1 {
    label="MDP 1 (P = 0.5)";
    style="rounded,filled";
    fillcolor="#FEF3C7";
    color="#F59E0B";
    penwidth=1.5;

    s0_1 [shape=box, style="rounded,filled", fillcolor="#FEF08A", color="#B45309", label="S0 (Stem)\n[MDP 1]"];
    s1_1 [shape=box, style="rounded,filled", fillcolor="#FEF08A", color="#B45309", label="S1 (Split)\n[MDP 1]"];
    s2_1 [shape=box, style="rounded,filled", fillcolor="#FEE2E2", color="#DC2626", label="S2 (Goal L)\nTerminal"];
    s3_1 [shape=box, style="rounded,filled", fillcolor="#DCFCE7", color="#059669", penwidth=2, label="S3 (Goal R)\nTerminal"];

    s0_1 -> s1_1 [label="Advance\n(Obs: O_1)", color="#D97706", penwidth=1.5];
    s1_1 -> s2_1 [label="Left\nR: 0.0", color="#94A3B8"];
    s1_1 -> s3_1 [label="Right\nR: +1.0", color="#059669", fontcolor="#059669", penwidth=2];
  }

  c_init -> s0_0 [label="0.50", color="#EA580C", penwidth=1.5];
  c_init -> s0_1 [label="0.50", color="#EA580C", penwidth=1.5];
}
"##;
    let pomdp_dot_path = out_dir.join("pomdp_structure.dot");
    let pomdp_svg_path = out_dir.join("pomdp_structure.svg");
    fs::write(&pomdp_dot_path, pomdp_dot)?;
    let parsed_pomdp = graphviz_rust::parse(pomdp_dot)?;
    let pomdp_svg_bytes = graphviz_rust::exec(
        parsed_pomdp,
        &mut graphviz_rust::printer::PrinterContext::default(),
        vec![graphviz_rust::cmd::CommandArg::Format(
            graphviz_rust::cmd::Format::Svg,
        )],
    )?;
    fs::write(&pomdp_svg_path, pomdp_svg_bytes)?;
    println!("  Generated MDP 0, MDP 1, and POMDP structure diagrams.");

    // -------------------------------------------------------------------------
    // 1. TEST 1: IDENTICAL OBSERVATIONS (BLIND Y-MAZE)
    // -------------------------------------------------------------------------
    println!("\n============================================================");
    println!(">>> [1] Running Test 1: Identical Observations (Blind)...   ");
    println!("============================================================");

    let dynamics1 = PomdpDynamics {
        obs_mode: ObsMode::Identical,
    };
    let mut sampler1 = RandomSampler::new(42);
    let model1 = NeutralModel;
    let selection1 = IsmctsSelection::<1>::new(1.414);
    let backup1 = VectorBackup::<1>::default();

    let mut tree1: TreeStore<Action, [f32; 1], IsmctsStats<1>, Observation> =
        TreeStore::with_capacity(20, 20, IsmctsStats::<1>::new());
    let root1 = tree1.insert_root(AgentId(0));

    let scheduler = IsmctsScheduler;
    scheduler.search(
        &mut tree1,
        &dynamics1,
        &model1,
        &selection1,
        &backup1,
        &mut sampler1,
        &(),
        root1,
        60,
    );

    println!("Completed 60 ISMCTS iterations on Test 1.");
    println!("  Allocated Nodes: {}", tree1.num_nodes());
    println!("  Allocated Edges: {}", tree1.num_edges());

    let opts1 = TreeVisualizerOptions::new()
        .with_highlight_pv(true)
        .with_render_chance_nodes(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_action_formatter(format_action)
        .with_delta_formatter(format_obs);

    write_svg_file(&tree1, root1, &opts1, out_dir.join("ismcts_test1_tree.svg"))?;
    write_png_file(&tree1, root1, &opts1, out_dir.join("ismcts_test1_tree.png"))?;
    write_dot_file(&tree1, root1, &opts1, out_dir.join("ismcts_test1_tree.dot"))?;

    println!("\n--- Test 1 Tree Statistics Breakdown ---");
    let mut test1_stats = Vec::new();
    for e in 0..tree1.num_edges() {
        let edge = EdgeId(e as u32);
        let action = tree1.edge_action(edge);
        let visits = tree1.stats.visits[e];
        let avail = tree1.stats.avail_visits[e];
        let q = tree1.stats.mean_value[e][0];
        let p = tree1.stats.priors[e];
        println!(
            "  Edge {:?} ({:?}): Visits={}, Avail={}, Q={:.3}, Prior={:.3}",
            edge, action, visits, avail, q, p
        );
        test1_stats.push((*action, visits, avail, q, p));
    }

    // -------------------------------------------------------------------------
    // 2. TEST 2: DISTINCT OBSERVATIONS (CUE-CONDITIONED Y-MAZE)
    // -------------------------------------------------------------------------
    println!("\n============================================================");
    println!(">>> [2] Running Test 2: Distinct Observations (Cued)...     ");
    println!("============================================================");

    let dynamics2 = PomdpDynamics {
        obs_mode: ObsMode::Distinct,
    };
    let mut sampler2 = RandomSampler::new(42);
    let model2 = NeutralModel;
    let selection2 = IsmctsSelection::<1>::new(1.414);
    let backup2 = VectorBackup::<1>::default();

    let mut tree2: TreeStore<Action, [f32; 1], IsmctsStats<1>, Observation> =
        TreeStore::with_capacity(30, 30, IsmctsStats::<1>::new());
    let root2 = tree2.insert_root(AgentId(0));

    scheduler.search(
        &mut tree2,
        &dynamics2,
        &model2,
        &selection2,
        &backup2,
        &mut sampler2,
        &(),
        root2,
        60,
    );

    println!("Completed 60 ISMCTS iterations on Test 2.");
    println!("  Allocated Nodes: {}", tree2.num_nodes());
    println!("  Allocated Edges: {}", tree2.num_edges());

    let opts2 = TreeVisualizerOptions::new()
        .with_highlight_pv(true)
        .with_render_chance_nodes(true)
        .with_rankdir(RankDir::TopToBottom)
        .with_action_formatter(format_action)
        .with_delta_formatter(format_obs);

    write_svg_file(&tree2, root2, &opts2, out_dir.join("ismcts_test2_tree.svg"))?;
    write_png_file(&tree2, root2, &opts2, out_dir.join("ismcts_test2_tree.png"))?;
    write_dot_file(&tree2, root2, &opts2, out_dir.join("ismcts_test2_tree.dot"))?;

    // Also export direct-edge rendering without virtual chance node
    let opts2_direct = TreeVisualizerOptions::new()
        .with_highlight_pv(true)
        .with_render_chance_nodes(false)
        .with_rankdir(RankDir::TopToBottom)
        .with_action_formatter(format_action)
        .with_delta_formatter(format_obs);
    write_svg_file(
        &tree2,
        root2,
        &opts2_direct,
        out_dir.join("ismcts_test2_tree_direct.svg"),
    )?;

    println!("\n--- Test 2 Tree Statistics Breakdown ---");
    let mut test2_stats = Vec::new();
    for e in 0..tree2.num_edges() {
        let edge = EdgeId(e as u32);
        let action = tree2.edge_action(edge);
        let visits = tree2.stats.visits[e];
        let avail = tree2.stats.avail_visits[e];
        let q = tree2.stats.mean_value[e][0];
        let p = tree2.stats.priors[e];
        println!(
            "  Edge {:?} ({:?}): Visits={}, Avail={}, Q={:.3}, Prior={:.3}",
            edge, action, visits, avail, q, p
        );
        test2_stats.push((*action, visits, avail, q, p));
    }

    // Verify mathematical properties:
    // Root edge (Advance):
    assert_eq!(tree1.stats.visits[0], 60);
    assert_eq!(tree2.stats.visits[0], 60);

    // In Test 1, both Left and Right have symmetric expected return ~ 0.500
    let left_q1 = tree1.stats.mean_value[1][0];
    let right_q1 = tree1.stats.mean_value[2][0];
    println!("\nTest 1 Invariant Verification:");
    println!("  Left Q = {left_q1:.3}, Right Q = {right_q1:.3} (Expected: ~0.500 each)");
    assert!(
        (left_q1 - 0.5).abs() <= 0.15,
        "Left Q ({left_q1}) should be approximately 0.500"
    );
    assert!(
        (right_q1 - 0.5).abs() <= 0.15,
        "Right Q ({right_q1}) should be approximately 0.500"
    );

    // In Test 2, tree branches into two distinct split nodes:
    let advance_branches: Vec<_> = tree2.delta_children(EdgeId(0)).collect();
    assert_eq!(
        advance_branches.len(),
        2,
        "Advance edge must branch into 2 observation subtrees"
    );

    let (node_o0, node_o1) = if *advance_branches[0].0 == Observation::Obs0 {
        (advance_branches[0].1, advance_branches[1].1)
    } else {
        (advance_branches[1].1, advance_branches[0].1)
    };

    println!("\nTest 2 Invariant Verification:");
    println!("  Subtree O_0 Node ID: {:?}", node_o0);
    println!("  Subtree O_1 Node ID: {:?}", node_o1);

    // Subtree O_0 edges:
    let o0_first_edge = tree2.first_child_edge(node_o0);
    let edge_o0_left = o0_first_edge;
    let edge_o0_right = EdgeId(o0_first_edge.0 + 1);

    let o0_left_q = tree2.stats.mean_value[edge_o0_left.as_usize()][0];
    let o0_left_v = tree2.stats.visits[edge_o0_left.as_usize()];
    let o0_right_q = tree2.stats.mean_value[edge_o0_right.as_usize()][0];
    let o0_right_v = tree2.stats.visits[edge_o0_right.as_usize()];

    println!(
        "  Subtree O_0 (MDP 0): Left Visits={}, Q={:.3} | Right Visits={}, Q={:.3}",
        o0_left_v, o0_left_q, o0_right_v, o0_right_q
    );
    assert_eq!(
        o0_left_q, 1.0,
        "In Subtree O_0, Left should have exact Q = 1.0"
    );
    assert_eq!(
        o0_right_q, 0.0,
        "In Subtree O_0, Right should have exact Q = 0.0"
    );
    assert!(
        o0_left_v > o0_right_v,
        "In Subtree O_0, Left should dominate visits"
    );

    // Subtree O_1 edges:
    let o1_first_edge = tree2.first_child_edge(node_o1);
    let edge_o1_left = o1_first_edge;
    let edge_o1_right = EdgeId(o1_first_edge.0 + 1);

    let o1_left_q = tree2.stats.mean_value[edge_o1_left.as_usize()][0];
    let o1_left_v = tree2.stats.visits[edge_o1_left.as_usize()];
    let o1_right_q = tree2.stats.mean_value[edge_o1_right.as_usize()][0];
    let o1_right_v = tree2.stats.visits[edge_o1_right.as_usize()];

    println!(
        "  Subtree O_1 (MDP 1): Left Visits={}, Q={:.3} | Right Visits={}, Q={:.3}",
        o1_left_v, o1_left_q, o1_right_v, o1_right_q
    );
    assert_eq!(
        o1_left_q, 0.0,
        "In Subtree O_1, Left should have exact Q = 0.0"
    );
    assert_eq!(
        o1_right_q, 1.0,
        "In Subtree O_1, Right should have exact Q = 1.0"
    );
    assert!(
        o1_right_v > o1_left_v,
        "In Subtree O_1, Right should dominate visits"
    );

    println!("\n>>> ALL MATHEMATICAL & STRUCTURAL INVARIANTS SATISFIED! <<<");

    // -------------------------------------------------------------------------
    // 3. GENERATE INTERACTIVE HTML REPORT
    // -------------------------------------------------------------------------
    let t1_left_v = tree1.stats.visits[1];
    let t1_left_q = tree1.stats.mean_value[1][0];
    let t1_right_v = tree1.stats.visits[2];
    let t1_right_q = tree1.stats.mean_value[2][0];
    let t1_avail = tree1.stats.avail_visits[1];

    generate_html_report(
        out_dir, t1_left_v, t1_left_q, t1_right_v, t1_right_q, t1_avail, o0_left_v, o0_left_q,
        o0_right_v, o0_right_q, o1_left_v, o1_left_q, o1_right_v, o1_right_q,
    )?;

    println!(
        "\nInteractive report written to: {}",
        out_dir.join("ismcts_pomdp_analysis.html").display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_html_report(
    out_dir: &Path,
    t1_l_v: u32,
    t1_l_q: f32,
    t1_r_v: u32,
    t1_r_q: f32,
    t1_avail: u32,
    o0_l_v: u32,
    o0_l_q: f32,
    o0_r_v: u32,
    o0_r_q: f32,
    o1_l_v: u32,
    o1_l_q: f32,
    o1_r_v: u32,
    o1_r_q: f32,
) -> io::Result<()> {
    let mdp0_svg = fs::read_to_string(out_dir.join("pomdp_mdp0.svg")).unwrap_or_default();
    let mdp1_svg = fs::read_to_string(out_dir.join("pomdp_mdp1.svg")).unwrap_or_default();
    let pomdp_struct_svg =
        fs::read_to_string(out_dir.join("pomdp_structure.svg")).unwrap_or_default();
    let test1_tree_svg =
        fs::read_to_string(out_dir.join("ismcts_test1_tree.svg")).unwrap_or_default();
    let test2_tree_svg =
        fs::read_to_string(out_dir.join("ismcts_test2_tree.svg")).unwrap_or_default();
    let test2_direct_svg =
        fs::read_to_string(out_dir.join("ismcts_test2_tree_direct.svg")).unwrap_or_default();
    let template = r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Single-Tree ISMCTS Validation: Y-Shaped POMDP</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 30px;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.6;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      background: #ffffff;
      padding: 32px 40px;
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    h1 {
      font-size: 28px;
      color: #0f172a;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 12px;
      margin-top: 0;
    }
    h2 {
      font-size: 20px;
      color: #1e293b;
      margin-top: 28px;
      border-left: 4px solid #2563eb;
      padding-left: 12px;
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-info { background: #dbeafe; color: #1e40af; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-top: 16px;
    }
    .card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
    }
    .card-header {
      font-weight: 700;
      font-size: 16px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .img-container {
      text-align: center;
      background: #f8fafc;
      padding: 16px;
      border-radius: 6px;
      border: 1px solid #f1f5f9;
      min-height: 260px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow-x: auto;
    }
    .img-container svg {
      max-width: 100%;
      height: auto;
      max-height: 480px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }
    th {
      background: #f8fafc;
      font-weight: 600;
      color: #475569;
    }
    .highlight-win {
      background: #f0fdf4;
      font-weight: 600;
      color: #15803d;
    }
    .highlight-lose {
      background: #fef2f2;
      color: #b91c1c;
    }
    .callout {
      background: #eff6ff;
      border-left: 4px solid #3b82f6;
      padding: 16px 20px;
      border-radius: 4px;
      margin: 20px 0;
    }
    .callout-title {
      font-weight: 700;
      color: #1d4ed8;
      margin-bottom: 6px;
    }
    code {
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Single-Tree ISMCTS Validation: Y-Shaped POMDP</h1>
    <p>
      This validation benchmarks the Single-Tree Information Set Monte Carlo Tree Search (Single-Tree ISMCTS)
      scheduler on a canonical Partially Observable Markov Decision Process (POMDP) composed of two hidden
      symmetric environments: <b>MDP 0</b> and <b>MDP 1</b>.
    </p>

    <div class="callout">
      <div class="callout-title">Core POMDP Mechanics & Information Set Conditioning</div>
      At the root (S_0), the agent has a uniform prior belief P(M=0) = P(M=1) = 0.50.
      In both worlds, the agent must first traverse the corridor via <code>Advance</code> to arrive at the decision junction (S_1).
      <ul>
        <li><b>MDP 0:</b> At S_1, choosing <code>Left</code> yields <b>+1.0</b> (win), while <code>Right</code> yields <b>0.0</b> (loss).</li>
        <li><b>MDP 1:</b> At S_1, choosing <code>Right</code> yields <b>+1.0</b> (win), while <code>Left</code> yields <b>0.0</b> (loss).</li>
      </ul>
    </div>

    <h2>1. Reference Environment Models</h2>
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <span>MDP 0 (Left Goal is Winning)</span>
          <span class="badge badge-info">P = 0.50</span>
        </div>
        <div class="img-container">
          __MDP0_SVG__
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <span>MDP 1 (Right Goal is Winning)</span>
          <span class="badge badge-warning">P = 0.50</span>
        </div>
        <div class="img-container">
          __MDP1_SVG__
        </div>
      </div>
    </div>

    <div class="card" style="margin-top: 20px;">
      <div class="card-header">
        <span>Combined POMDP State Transition Graph</span>
        <span class="badge badge-success">Bipartite Model</span>
      </div>
      <div class="img-container">
        __POMDP_STRUCT_SVG__
      </div>
    </div>

    <h2>2. Experimental Results Comparison</h2>
    <div class="grid-2">
      <!-- Test 1 Card -->
      <div class="card">
        <div class="card-header">
          <span>Test 1: Identical Observations (Blind)</span>
          <span class="badge badge-warning">Unresolved Decision</span>
        </div>
        <div class="img-container">
          __TEST1_TREE_SVG__
        </div>
        <p style="font-size: 13px; color: #64748b; margin-top: 10px;">
          Both worlds emit the exact same observation <code>Uniform</code> upon stepping <code>Advance</code>.
          The agent pools all determinizations into a single shared information set node.
        </p>
        <table>
          <thead>
            <tr>
              <th>Action at Split</th>
              <th>Visits (N)</th>
              <th>Avail Visits</th>
              <th>Empirical Q</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>Left</code></td>
              <td>__T1_L_V__</td>
              <td>__T1_AVAIL__</td>
              <td>__T1_L_Q__</td>
              <td>Balanced (~0.500)</td>
            </tr>
            <tr>
              <td><code>Right</code></td>
              <td>__T1_R_V__</td>
              <td>__T1_AVAIL__</td>
              <td>__T1_R_Q__</td>
              <td>Balanced (~0.500)</td>
            </tr>
          </tbody>
        </table>
        <p style="font-size: 13px; color: #475569; margin-top: 8px;">
          <b>Result:</b> Because no cue is available, E[Q(Left)] = E[Q(Right)] &approx; 0.500.
          ISMCTS balances visits symmetrically, correctly preserving Bayesian uncertainty under ignorance.
        </p>
      </div>

      <!-- Test 2 Card -->
      <div class="card">
        <div class="card-header">
          <span>Test 2: Distinct Observations (Cued)</span>
          <span class="badge badge-success">Optimal Policy Discovered</span>
        </div>
        <div class="img-container">
          __TEST2_TREE_SVG__
        </div>
        <p style="font-size: 13px; color: #64748b; margin-top: 10px;">
          Stepping <code>Advance</code> emits O_0 in MDP 0 and O_1 in MDP 1.
          Single-Tree ISMCTS automatically splits at the chance node into two observation subtrees!
        </p>
        <table>
          <thead>
            <tr>
              <th>Observation Subtree</th>
              <th>Action</th>
              <th>Visits (N)</th>
              <th>Empirical Q</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight-win">
              <td><b>Subtree O_0 (MDP 0)</b></td>
              <td><code>Left</code></td>
              <td>__O0_L_V__</td>
              <td>__O0_L_Q__</td>
              <td><b>Optimal Choice (PV)</b></td>
            </tr>
            <tr class="highlight-lose">
              <td>Subtree O_0 (MDP 0)</td>
              <td><code>Right</code></td>
              <td>__O0_R_V__</td>
              <td>__O0_R_Q__</td>
              <td>Suboptimal</td>
            </tr>
            <tr class="highlight-lose">
              <td>Subtree O_1 (MDP 1)</td>
              <td><code>Left</code></td>
              <td>__O1_L_V__</td>
              <td>__O1_L_Q__</td>
              <td>Suboptimal</td>
            </tr>
            <tr class="highlight-win">
              <td><b>Subtree O_1 (MDP 1)</b></td>
              <td><code>Right</code></td>
              <td>__O1_R_V__</td>
              <td>__O1_R_Q__</td>
              <td><b>Optimal Choice (PV)</b></td>
            </tr>
          </tbody>
        </table>
        <p style="font-size: 13px; color: #475569; margin-top: 8px;">
          <b>Result:</b> Conditioned on observation cues, Single-Tree ISMCTS isolates the deterministic sub-dynamics.
          PUCT concentrates almost all rollouts on the winning branch in each subtree:
          &pi;*(O_0) = Left and &pi;*(O_1) = Right.
        </p>
      </div>
    </div>

    <h2>3. Direct-Edge Rendering (Alternative Visual View)</h2>
    <div class="card">
      <div class="card-header">
        <span>Test 2 Rendered with Direct Transition Edges (No Virtual Chance Node)</span>
      </div>
      <div class="img-container">
        __TEST2_DIRECT_SVG__
      </div>
    </div>
  </div>
</body>
</html>
"##;

    let html = template
        .replace("__MDP0_SVG__", &mdp0_svg)
        .replace("__MDP1_SVG__", &mdp1_svg)
        .replace("__POMDP_STRUCT_SVG__", &pomdp_struct_svg)
        .replace("__TEST1_TREE_SVG__", &test1_tree_svg)
        .replace("__TEST2_TREE_SVG__", &test2_tree_svg)
        .replace("__TEST2_DIRECT_SVG__", &test2_direct_svg)
        .replace("__T1_L_V__", &t1_l_v.to_string())
        .replace("__T1_L_Q__", &format!("{t1_l_q:.3}"))
        .replace("__T1_R_V__", &t1_r_v.to_string())
        .replace("__T1_R_Q__", &format!("{t1_r_q:.3}"))
        .replace("__T1_AVAIL__", &t1_avail.to_string())
        .replace("__O0_L_V__", &o0_l_v.to_string())
        .replace("__O0_L_Q__", &format!("{o0_l_q:.3}"))
        .replace("__O0_R_V__", &o0_r_v.to_string())
        .replace("__O0_R_Q__", &format!("{o0_r_q:.3}"))
        .replace("__O1_L_V__", &o1_l_v.to_string())
        .replace("__O1_L_Q__", &format!("{o1_l_q:.3}"))
        .replace("__O1_R_V__", &o1_r_v.to_string())
        .replace("__O1_R_Q__", &format!("{o1_r_q:.3}"));

    let mut f = fs::File::create(out_dir.join("ismcts_pomdp_analysis.html"))?;
    f.write_all(html.as_bytes())?;
    Ok(())
}

#[test]
fn test_run_ismcts_pomdp_experiment() {
    main().expect("ismcts_pomdp experiment failed");
}
