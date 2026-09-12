//! Unit and integration tests for the 2048 game engine and Expectimax chance-node MCTS.

use crate::agent::{Agent, HeuristicAgent, MctsAgent, RandomAgent};
use crate::dynamics::Tzf8Dynamics;
use crate::evaluator::{CornerHeuristicEvaluator, UniformEvaluator};
use crate::game::{Direction, TileSpawn, Tzf8State};
use crate::world::Tzf8World;
use mcts_engine::backup::SingleAgentBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::{EdgeId, TreeStore};
use mcts_traits::{AgentId, World};

#[test]
fn test_shift_left_and_merge() {
    let mut state = Tzf8State::new_empty();
    // Row 0: [2, 2, 0, 0] -> [4, 0, 0, 0] (score +4)
    state.board[0] = [2, 2, 0, 0];
    // Row 1: [2, 2, 2, 2] -> [4, 4, 0, 0] (score +8)
    state.board[1] = [2, 2, 2, 2];
    // Row 2: [2, 0, 2, 4] -> [4, 4, 0, 0] (score +4)
    state.board[2] = [2, 0, 2, 4];
    // Row 3: [0, 0, 0, 0] -> [0, 0, 0, 0] (score +0)
    state.board[3] = [0, 0, 0, 0];

    let (changed, score_gain) = state.move_board(Direction::Left);
    assert!(changed);
    assert_eq!(score_gain, 16);
    assert_eq!(state.board[0], [4, 0, 0, 0]);
    assert_eq!(state.board[1], [4, 4, 0, 0]);
    assert_eq!(state.board[2], [4, 4, 0, 0]);
    assert_eq!(state.board[3], [0, 0, 0, 0]);
}

#[test]
fn test_shift_right_up_down() {
    let mut state = Tzf8State::new_empty();
    state.board[0] = [2, 0, 0, 2];
    let (changed, score) = state.move_board(Direction::Right);
    assert!(changed);
    assert_eq!(score, 4);
    assert_eq!(state.board[0], [0, 0, 0, 4]);

    // Up shift
    let mut state_up = Tzf8State::new_empty();
    state_up.board[0][1] = 4;
    state_up.board[2][1] = 4;
    let (changed, score) = state_up.move_board(Direction::Up);
    assert!(changed);
    assert_eq!(score, 8);
    assert_eq!(state_up.board[0][1], 8);
    assert_eq!(state_up.board[2][1], 0);

    // Down shift
    let mut state_down = Tzf8State::new_empty();
    state_down.board[0][2] = 8;
    state_down.board[1][2] = 8;
    let (changed, score) = state_down.move_board(Direction::Down);
    assert!(changed);
    assert_eq!(score, 16);
    assert_eq!(state_down.board[3][2], 16);
}

#[test]
fn test_terminal_detection() {
    let mut state = Tzf8State::new_empty();
    // Checkerboard pattern with no adjacent identical tiles
    state.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]];
    assert!(!state.can_move());
    let mut legal = Vec::new();
    state.legal_directions(&mut legal);
    assert!(legal.is_empty());
}

#[test]
fn test_world_play_to_completion() {
    let world = Tzf8World::with_seed(42);
    let mut state = world.initial();
    let mut random = RandomAgent::default();

    let mut moves = 0;
    while !world.is_terminal(&state) && moves < 2000 {
        let act = random.select_action(&state);
        let outcome = world.step_action(&mut state, act);
        moves += 1;
        if outcome.terminated {
            break;
        }
    }

    assert!(moves > 0);
    assert!(state.score > 0);
}

#[test]
fn test_chance_node_branching_expectimax() {
    // Crucial architectural test:
    // Verify that MCTS over Tzf8Dynamics creates distinct delta-branches
    // for different stochastic tile spawn events under the same action edge!
    let dynamics = Tzf8Dynamics::with_seed(12345);
    let mut state = Tzf8State::new_empty();
    state.board[0] = [2, 0, 0, 0];
    state.board[1] = [2, 0, 0, 0];

    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.414 };
    let backup = SingleAgentBackup::new(1.0);
    let scheduler = SequentialScheduler;
    let model = UniformEvaluator;

    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<Direction, [f32; 1], _, Option<TileSpawn>> =
        TreeStore::with_capacity(500, 500, stats);

    let root = tree.insert_root(AgentId(0));
    scheduler.search(
        &mut tree, &dynamics, &model, &selection, &backup, root, &state,
        200, // 200 simulation sweeps
    );

    assert!(tree.num_nodes() > 2);
    let num_children = tree.num_children(root);
    assert!(num_children > 0);

    let first_edge = tree.first_child_edge(root);
    let mut max_delta_branches = 0;

    for i in 0..num_children {
        let edge = EdgeId(first_edge.0 + i);
        let branches: Vec<_> = tree.delta_children(edge).collect();
        if branches.len() > max_delta_branches {
            max_delta_branches = branches.len();
        }
    }

    // In 200 iterations on a board with 14 empty cells, the root action edge MUST
    // have branched into multiple distinct TileSpawns!
    assert!(
        max_delta_branches > 1,
        "Expected stochastic chance branching under action edge, but got max branches: {max_delta_branches}"
    );
}

#[test]
fn test_heuristic_agent_eval() {
    let mut agent = HeuristicAgent::default();
    let world = Tzf8World::with_seed(999);
    let state = world.initial();
    let action = agent.select_action(&state);
    assert!(Direction::ALL.contains(&action));
}

#[test]
fn test_mcts_agent_eval() {
    let mut agent = MctsAgent::new(
        "MCTS-Test",
        50,
        1.414,
        1.0,
        CornerHeuristicEvaluator::new(),
        false,
    );
    let world = Tzf8World::with_seed(101);
    let state = world.initial();
    let action = agent.select_action(&state);
    assert!(Direction::ALL.contains(&action));
}
