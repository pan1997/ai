//! Comprehensive unit and integration tests for `mcts-envs` games and evaluators.

use crate::evaluators::{RolloutEvaluator, UniformRandomModel};
use crate::hex::{HexPlayer, HexState, HexWorld};
use crate::kuhn_poker::{KuhnAction, KuhnAgentDynamics, KuhnWorld};
use crate::tzf8::{Direction, Tzf8Dynamics, Tzf8State};
use mcts_engine::backup::{SingleAgentBackup, VectorBackup};
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentDynamics, AgentId, Model, TurnBasedDynamics, World};

#[test]
fn test_hex_win_detection() {
    let env = TurnBasedDynamics::new(HexWorld::<3>::new());
    let mut state = HexState::<3>::new();

    // Black connects top to bottom (row 0 to row 2) along column 0
    AgentDynamics::step(&env, &mut state, &0); // Black (0,0)
    AgentDynamics::step(&env, &mut state, &1); // White (0,1)
    AgentDynamics::step(&env, &mut state, &3); // Black (1,0)
    AgentDynamics::step(&env, &mut state, &4); // White (1,1)
    let t = AgentDynamics::step(&env, &mut state, &6); // Black (2,0) -> connected Top to Bottom!
    assert!(t.terminated);
    assert_eq!(t.reward, [1.0, -1.0]);
}

#[test]
fn test_hex_white_win_detection() {
    let env = TurnBasedDynamics::new(HexWorld::<3>::new());
    let mut state = HexState::<3>::new();

    // White connects Left to Right along row 0: (0,0), (0,1), (0,2) = cells 0, 1, 2
    AgentDynamics::step(&env, &mut state, &3); // Black (1,0)
    AgentDynamics::step(&env, &mut state, &0); // White (0,0)
    AgentDynamics::step(&env, &mut state, &4); // Black (1,1)
    AgentDynamics::step(&env, &mut state, &1); // White (0,1)
    AgentDynamics::step(&env, &mut state, &5); // Black (1,2)

    let t = AgentDynamics::step(&env, &mut state, &2); // White (0,2) -> White connects Left to Right!
    assert!(t.terminated);
    assert_eq!(t.reward, [-1.0, 1.0]); // White wins (+1 for White, -1 for Black)
}

#[test]
fn test_hex_world_interface() {
    let world = HexWorld::<3>::new();
    let mut ws = World::initial(&world);

    assert_eq!(World::n_players(&world), 2);
    let mut acts = Vec::new();
    World::actions(&world, &ws, 0, &mut acts);
    assert_eq!(acts.len(), 9);
    World::actions(&world, &ws, 1, &mut acts);
    assert_eq!(acts.len(), 0);

    let (rewards, terminated) = World::step(&world, &mut ws, &[0, 0]);
    assert!(!terminated);
    assert_eq!(rewards, vec![0.0, 0.0]);
    assert_eq!(ws.current_player, HexPlayer::White);
}

#[test]
fn test_hex_mcts_search() {
    let env = TurnBasedDynamics::new(HexWorld::<3>::new());
    let model = UniformRandomModel::new(env, 2);
    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.4 };
    let backup = VectorBackup::<2>::default();
    let stats = MultiAgentPuctStats::<2>::new();

    let mut tree = TreeStore::with_capacity(50, 50, stats);
    let root = tree.insert_root(AgentId(0));
    let initial_state = AgentDynamics::initial(&env);

    let scheduler = SequentialScheduler;
    scheduler.search(&mut tree, &env, &model, &selection, &backup, root, &initial_state, 10);

    // Root should have 9 children (cells 0..9)
    assert_eq!(tree.num_children(root), 9);
    let total_visits: u32 = tree.child_edges(root).map(|e| tree.stats.visits[e.as_usize()]).sum();
    assert_eq!(total_visits, 10);
}

#[test]
fn test_hex_dsu_deep_connectivity_and_path_compression() {
    let mut state = HexState::<5>::new();

    // Verify initial states are not won
    assert!(!state.is_won(HexPlayer::Black));
    assert!(!state.is_won(HexPlayer::White));

    // Construct a zigzag path for Black connecting top row (r=0) to bottom row (r=4):
    // (0, 2) -> (1, 2) -> (2, 1) -> (3, 1) -> (4, 1)
    let moves = [
        (0, 2),
        (1, 2),
        (2, 1),
        (3, 1),
        (4, 1),
    ];

    for (i, &(r, c)) in moves.iter().enumerate() {
        let idx = HexState::<5>::idx(r, c);
        state.current_player = HexPlayer::Black;
        let won = state.play_move(idx);
        if i < moves.len() - 1 {
            assert!(!won, "Move {i} at ({r},{c}) should not complete the path yet");
            assert!(!state.is_won(HexPlayer::Black));
        } else {
            assert!(won, "Final move should complete the top-to-bottom winning chain");
            assert!(state.is_won(HexPlayer::Black));
            assert!(state.check_win(HexPlayer::Black));
        }
    }
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
fn test_tzf8_sliding_and_merging_mechanics() {
    let mut state = Tzf8State::new(42);
    // Explicit tile configuration
    state.board = [
        [2, 2, 4, 4], // Left slide -> [4, 8, 0, 0], gain = 4 + 8 = 12
        [2, 2, 2, 0], // Left slide -> [4, 2, 0, 0], gain = 4 (no double merge)
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ];

    let (changed, score_gain) = state.move_board(Direction::Left);
    assert!(changed);
    assert_eq!(score_gain, 16);
    assert_eq!(state.board[0], [4, 8, 0, 0]);
    assert_eq!(state.board[1], [4, 2, 0, 0]);

    // Test locked board where no moves are possible
    state.board = [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ];
    assert!(!state.can_move());
}

#[test]
fn test_kuhn_poker_world_and_agent_dynamics() {
    let world = KuhnWorld { fixed_deal: Some([2, 0]) }; // P0 has King (2), P1 has Jack (0)
    let mut ws = world.initial();
    let p0_obs = world.observe(&ws, 0);
    assert_eq!(p0_obs.my_card, 2);

    // P0 bets with King
    let mut acts = Vec::new();
    world.actions(&ws, 0, &mut acts);
    assert!(acts.contains(&KuhnAction::Bet));

    let (_, done) = world.step(&mut ws, &[KuhnAction::Bet, KuhnAction::Check]);
    assert!(!done);

    // P1 folds
    let (rewards, done_final) = world.step(&mut ws, &[KuhnAction::Check, KuhnAction::Fold]);
    assert!(done_final);
    assert_eq!(rewards, vec![1.0, -1.0]);

    // Test agent internal dynamics for Kuhn
    let agent_dyn = KuhnAgentDynamics {
        my_card: 2, // King
        opponent_call_rate: 0.8,
    };
    let mut s0 = agent_dyn.initial();
    let t = agent_dyn.step(&mut s0, &KuhnAction::Bet);
    assert!(t.terminated);
    assert_eq!(t.reward[0], 2.0); // Opponent called and King won 2 chips!
}

#[test]
fn test_kuhn_poker_game_tree_showdowns() {
    let world_check_check = KuhnWorld { fixed_deal: Some([2, 1]) }; // P0 King, P1 Queen
    let mut ws = world_check_check.initial();
    // Step 1: P0 Checks
    let (_, done_p0) = world_check_check.step(&mut ws, &[KuhnAction::Check, KuhnAction::Check]);
    assert!(!done_p0);
    // Step 2: P1 Checks -> Showdown
    let (rewards, done) = world_check_check.step(&mut ws, &[KuhnAction::Check, KuhnAction::Check]);
    assert!(done);
    assert_eq!(rewards, vec![1.0, -1.0]); // King beats Queen, wins 1 ante

    let world_bet_call = KuhnWorld { fixed_deal: Some([0, 2]) }; // P0 Jack, P1 King
    let mut ws2 = world_bet_call.initial();
    // P0 Bet, P1 passes
    let (_, _) = world_bet_call.step(&mut ws2, &[KuhnAction::Bet, KuhnAction::Check]);
    // P1 Calls
    let (rewards2, done2) = world_bet_call.step(&mut ws2, &[KuhnAction::Check, KuhnAction::Call]);
    assert!(done2);
    assert_eq!(rewards2, vec![-2.0, 2.0]); // King beats Jack, wins 2 chips
}

#[test]
fn test_evaluators_rollout_and_random_edge_cases() {
    let env = TurnBasedDynamics::new(HexWorld::<3>::new());

    // 1. RolloutEvaluator with max_depth = 0
    let rollout = RolloutEvaluator::new(env, 2, 0);
    let s = HexState::<3>::new();
    let eval = rollout.evaluate(&s);
    assert_eq!(eval.priors.len(), 9);
    assert_eq!(eval.values.len(), 1);

    // 2. UniformRandomModel on a full board (0 actions)
    let random_model = UniformRandomModel::new(env, 2);
    let mut full_state = HexState::<3>::new();
    full_state.board = vec![Some(HexPlayer::Black); 9];
    let eval_full = random_model.evaluate(&full_state);
    assert!(eval_full.priors.is_empty());
    assert_eq!(eval_full.values, vec![0.0, 0.0]);
}
