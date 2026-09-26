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

#[test]
fn test_ismcts_y_shaped_pomdp_blind_and_cued() {
    use mcts_engine::backup::VectorBackup;
    use mcts_engine::scheduler::IsmctsScheduler;
    use mcts_engine::selection::ismcts::{IsmctsSelection, IsmctsStats};
    use mcts_traits::belief::BeliefSampler;
    use mcts_traits::{ActionModel, AgentDynamics, AgentId, Evaluation, Model, StepOutcome};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Action {
        Advance = 0,
        Left = 1,
        Right = 2,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Step {
        Stem,
        Split,
        Terminal,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct State {
        mdp_id: u8,
        step: Step,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    enum Obs {
        #[default]
        Same,
        O0,
        O1,
    }

    struct Pomdp {
        cued: bool,
    }

    impl AgentDynamics for Pomdp {
        type State = State;
        type Action = Action;
        type Reward = [f32; 1];
        type StepDelta = Obs;

        fn initial(&self) -> State {
            State {
                mdp_id: 0,
                step: Step::Stem,
            }
        }

        fn actions(&self, state: &State, out: &mut Vec<Action>) {
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

        fn expand_actions(&self, state: &State, out: &mut Vec<Action>) {
            self.actions(state, out);
        }

        fn current_agent(&self, _state: &State) -> AgentId {
            AgentId(0)
        }

        fn step(&self, state: &mut State, action: &Action) -> StepOutcome<[f32; 1], Obs> {
            match state.step {
                Step::Stem => {
                    state.step = Step::Split;
                    let delta = if self.cued {
                        if state.mdp_id == 0 { Obs::O0 } else { Obs::O1 }
                    } else {
                        Obs::Same
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
                        Action::Advance => panic!(),
                    };
                    StepOutcome::with_delta(reward, Obs::Same, true)
                }
                Step::Terminal => panic!(),
            }
        }
    }

    struct Sampler {
        rng: StdRng,
    }

    impl BeliefSampler for Sampler {
        type State = State;
        type Context = ();

        fn sample(&mut self, _ctx: &()) -> State {
            State {
                mdp_id: if self.rng.gen_bool(0.5) { 0 } else { 1 },
                step: Step::Stem,
            }
        }
    }

    struct Neutral;
    impl Model<State> for Neutral {
        fn evaluate(&self, _s: &State) -> Evaluation {
            Evaluation {
                priors: vec![0.5, 0.5],
                values: vec![0.0],
            }
        }
    }
    impl ActionModel<State, Action> for Neutral {
        fn evaluate_actions(&self, s: &State, acts: &[Action]) -> Evaluation {
            let n = acts.len();
            let mut eval = self.evaluate(s);
            eval.priors = if n > 0 {
                vec![1.0 / n as f32; n]
            } else {
                vec![]
            };
            eval
        }
    }

    let scheduler = IsmctsScheduler;
    let selection = IsmctsSelection::<1>::new(1.414);
    let backup = VectorBackup::<1>::default();

    // 1. Blind Test
    {
        let dynamics = Pomdp { cued: false };
        let mut sampler = Sampler {
            rng: StdRng::seed_from_u64(42),
        };
        let mut tree: TreeStore<Action, [f32; 1], IsmctsStats<1>, Obs> =
            TreeStore::with_capacity(20, 20, IsmctsStats::<1>::new());
        let root = tree.insert_root(AgentId(0));

        scheduler.search(
            &mut tree,
            &dynamics,
            &Neutral,
            &selection,
            &backup,
            &mut sampler,
            &(),
            root,
            60,
        );

        // One advance edge, one split node with Left & Right
        assert_eq!(tree.stats.visits[0], 60);
        let q_left = tree.stats.mean_value[1][0];
        let q_right = tree.stats.mean_value[2][0];
        assert!((q_left - 0.5).abs() <= 0.15, "q_left: {q_left}");
        assert!((q_right - 0.5).abs() <= 0.15, "q_right: {q_right}");
    }

    // 2. Cued Test
    {
        let dynamics = Pomdp { cued: true };
        let mut sampler = Sampler {
            rng: StdRng::seed_from_u64(42),
        };
        let mut tree: TreeStore<Action, [f32; 1], IsmctsStats<1>, Obs> =
            TreeStore::with_capacity(30, 30, IsmctsStats::<1>::new());
        let root = tree.insert_root(AgentId(0));

        scheduler.search(
            &mut tree,
            &dynamics,
            &Neutral,
            &selection,
            &backup,
            &mut sampler,
            &(),
            root,
            60,
        );

        let branches: Vec<_> = tree.delta_children(EdgeId(0)).collect();
        assert_eq!(branches.len(), 2);

        let (node_o0, node_o1) = if *branches[0].0 == Obs::O0 {
            (branches[0].1, branches[1].1)
        } else {
            (branches[1].1, branches[0].1)
        };

        // In Subtree O0 (MDP 0): Left wins (Q=1.0), Right loses (Q=0.0)
        let o0_edge0 = tree.first_child_edge(node_o0);
        let o0_q_left = tree.stats.mean_value[o0_edge0.as_usize()][0];
        let o0_v_left = tree.stats.visits[o0_edge0.as_usize()];
        let o0_q_right = tree.stats.mean_value[o0_edge0.as_usize() + 1][0];
        let o0_v_right = tree.stats.visits[o0_edge0.as_usize() + 1];

        assert_eq!(o0_q_left, 1.0);
        assert_eq!(o0_q_right, 0.0);
        assert!(o0_v_left > o0_v_right);

        // In Subtree O1 (MDP 1): Left loses (Q=0.0), Right wins (Q=1.0)
        let o1_edge0 = tree.first_child_edge(node_o1);
        let o1_q_left = tree.stats.mean_value[o1_edge0.as_usize()][0];
        let o1_v_left = tree.stats.visits[o1_edge0.as_usize()];
        let o1_q_right = tree.stats.mean_value[o1_edge0.as_usize() + 1][0];
        let o1_v_right = tree.stats.visits[o1_edge0.as_usize() + 1];

        assert_eq!(o1_q_left, 0.0);
        assert_eq!(o1_q_right, 1.0);
        assert!(o1_v_right > o1_v_left);
    }
}
