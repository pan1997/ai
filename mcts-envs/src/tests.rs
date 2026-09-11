use crate::connect4::{Connect4Dynamics, Connect4State};
use crate::evaluators::{RolloutEvaluator, UniformRandomModel};
use crate::hex::{HexDynamics, HexState};
use crate::kuhn_poker::{KuhnAction, KuhnAgentDynamics, KuhnWorld};
use crate::tzf8::{Tzf8Dynamics, Tzf8State};
use mcts_engine::backup::{SingleAgentBackup, VectorBackup};
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentDynamics, AgentId, World};

#[test]
fn test_connect4_win_detection() {
    let env = Connect4Dynamics::<4, 4>;
    let mut state = Connect4State::<4, 4>::new();

    // Red in col 0, Yellow in col 1, repeated until Red has 4 vertical checkers
    state = AgentDynamics::step(&env, state, &0).next_state; // Red
    state = AgentDynamics::step(&env, state, &1).next_state; // Yellow
    state = AgentDynamics::step(&env, state, &0).next_state; // Red
    state = AgentDynamics::step(&env, state, &1).next_state; // Yellow
    state = AgentDynamics::step(&env, state, &0).next_state; // Red
    state = AgentDynamics::step(&env, state, &1).next_state; // Yellow

    let t = AgentDynamics::step(&env, state, &0); // Red wins!
    assert!(t.terminated);
    assert_eq!(t.reward, [1.0, -1.0]);
}

#[test]
fn test_connect4_mcts_search() {
    let env = Connect4Dynamics::<4, 4>;
    let model = UniformRandomModel::new(env, 2);
    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.4 };
    let backup = VectorBackup::<2>::default();
    let stats = MultiAgentPuctStats::<2>::new();

    let mut tree = TreeStore::with_capacity(50, 50, stats);
    let root = tree.insert_root(AgentId(0));
    let initial_state = AgentDynamics::initial(&env);

    let scheduler = SequentialScheduler;
    scheduler.search(&mut tree, &env, &model, &selection, &backup, root, &initial_state, 10);

    // Root should have 4 children (columns 0..4)
    assert_eq!(tree.num_children(root), 4);
    let total_visits: u32 = tree.child_edges(root).map(|e| tree.stats.visits[e.as_usize()]).sum();
    assert_eq!(total_visits, 10);
}

#[test]
fn test_hex_win_detection() {
    let env = HexDynamics::<3>;
    let mut state = HexState::<3>::new();

    // Black connects top to bottom (row 0 to row 2) along column 0
    state = AgentDynamics::step(&env, state, &0).next_state; // Black (0,0)
    state = AgentDynamics::step(&env, state, &1).next_state; // White (0,1)
    state = AgentDynamics::step(&env, state, &3).next_state; // Black (1,0)
    state = AgentDynamics::step(&env, state, &4).next_state; // White (1,1)
    let t = AgentDynamics::step(&env, state, &6);            // Black (2,0) -> connected Top to Bottom!
    assert!(t.terminated);
    assert_eq!(t.reward, [1.0, -1.0]);
}

#[test]
fn test_tzf8_mcts_search() {
    let env = Tzf8Dynamics;
    let model = RolloutEvaluator::new(env, 5, 10);
    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.0 };
    let backup = SingleAgentBackup::new(0.99);
    let stats = MultiAgentPuctStats::<1>::new();

    let mut tree = TreeStore::with_capacity(30, 30, stats);
    let root = tree.insert_root(AgentId(0));
    let state = Tzf8State::new(42);

    let scheduler = SequentialScheduler;
    scheduler.search(&mut tree, &env, &model, &selection, &backup, root, &state, 10);

    assert!(tree.num_children(root) > 0);
    let total_visits: u32 = tree.child_edges(root).map(|e| tree.stats.visits[e.as_usize()]).sum();
    assert_eq!(total_visits, 10);
}

#[test]
fn test_kuhn_poker_world_and_agent_dynamics() {
    let world = KuhnWorld { fixed_deal: Some([2, 0]) }; // P0 has King (2), P1 has Jack (0)
    let ws = world.initial();
    let p0_obs = world.observe(&ws, 0);
    assert_eq!(p0_obs.my_card, 2);

    // P0 bets with King
    let p0_actions = world.actions(&ws, 0);
    assert!(p0_actions.contains(&KuhnAction::Bet));

    let (ws_after_bet, _, done) = world.step(ws, &[KuhnAction::Bet, KuhnAction::Check]);
    assert!(!done);

    // P1 folds
    let (_final_ws, rewards, done_final) = world.step(ws_after_bet, &[KuhnAction::Check, KuhnAction::Fold]);
    assert!(done_final);
    assert_eq!(rewards, vec![1.0, -1.0]);

    // Test agent internal dynamics for Kuhn
    let agent_dyn = KuhnAgentDynamics {
        my_card: 2, // King
        opponent_call_rate: 0.8,
    };
    let s0 = agent_dyn.initial();
    let t = agent_dyn.step(s0, &KuhnAction::Bet);
    assert!(t.terminated);
    assert_eq!(t.reward[0], 2.0); // Opponent called and King won 2 chips!
}

