//! Comprehensive unit and integration tests for Connect 4 game engine, dynamics, and MCTS agents.

use crate::agent::{Agent, MctsAgent, RandomAgent};
use crate::dynamics::Connect4Dynamics;
use crate::evaluator::{RolloutEvaluator, UniformEvaluator};
use crate::game::{Connect4State, Player};
use crate::render::{render_board, render_board_styled};
use mcts_traits::{AgentDynamics, Model, World};

#[test]
fn test_initial_state_empty() {
    let state = Connect4State::<6, 7>::new();
    assert_eq!(state.current_player, Player::Red);
    assert!(!state.is_board_full());
    for col in 0..7 {
        assert!(!state.is_column_full(col));
        for row in 0..6 {
            assert_eq!(state.board[row][col], None);
        }
    }
    let mut actions = Vec::new();
    state.legal_actions(&mut actions);
    assert_eq!(actions, vec![0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_drop_piece_stacking() {
    let mut state = Connect4State::<6, 7>::new();
    // Drop in col 3
    let r1 = state.drop_piece(3).unwrap();
    assert_eq!(r1, 5); // bottom row
    assert_eq!(state.board[5][3], Some(Player::Red));

    state.current_player = Player::Yellow;
    let r2 = state.drop_piece(3).unwrap();
    assert_eq!(r2, 4); // one row above
    assert_eq!(state.board[4][3], Some(Player::Yellow));

    // Fill the rest of col 3
    for _ in 0..4 {
        state.drop_piece(3).unwrap();
    }
    assert!(state.is_column_full(3));

    // Attempting to drop into full col 3 must fail
    assert!(state.drop_piece(3).is_err());
    // Out of bounds col must fail
    assert!(state.drop_piece(7).is_err());
}

#[test]
fn test_horizontal_win() {
    let mut state = Connect4State::<6, 7>::new();
    // Red plays cols 0, 1, 2, 3 on bottom row
    for col in 0..4 {
        state.current_player = Player::Red;
        let r = state.drop_piece(col).unwrap();
        if col < 3 {
            assert!(!state.check_win_at(r, col, Player::Red));
        } else {
            assert!(state.check_win_at(r, col, Player::Red));
        }
    }
    assert!(state.has_won(Player::Red));
    assert!(!state.has_won(Player::Yellow));
}

#[test]
fn test_vertical_win() {
    let mut state = Connect4State::<6, 7>::new();
    // Yellow plays 4 pieces vertically in col 2
    state.current_player = Player::Yellow;
    for i in 0..4 {
        let r = state.drop_piece(2).unwrap();
        if i < 3 {
            assert!(!state.check_win_at(r, 2, Player::Yellow));
        } else {
            assert!(state.check_win_at(r, 2, Player::Yellow));
        }
    }
    assert!(state.has_won(Player::Yellow));
    assert!(!state.has_won(Player::Red));
}

#[test]
fn test_diagonal_down_right_win() {
    let mut state = Connect4State::<6, 7>::new();
    // Create diagonal down-right: (2, 0), (3, 1), (4, 2), (5, 3)
    // Board has rows 0..5 (5 is bottom)
    // Col 0: needs 3 pieces to reach row 3? Bottom is 5, then 4, 3, 2.
    // For cell (5, 0): bottom of col 0.
    // (4, 1): 2 pieces in col 1.
    // (3, 2): 3 pieces in col 2.
    // (2, 3): 4 pieces in col 3.
    // This forms an up-right diagonal from (5, 0) to (2, 3).
    // Let's test diagonal down-right from (2, 0) to (5, 3):
    // (2, 0) needs 4 pieces in col 0.
    // (3, 1) needs 3 pieces in col 1.
    // (4, 2) needs 2 pieces in col 2.
    // (5, 3) needs 1 piece in col 3.

    // Col 0: 3 dummy pieces, then Red
    state.board[5][0] = Some(Player::Yellow);
    state.board[4][0] = Some(Player::Yellow);
    state.board[3][0] = Some(Player::Yellow);
    state.board[2][0] = Some(Player::Red);

    // Col 1: 2 dummy pieces, then Red
    state.board[5][1] = Some(Player::Yellow);
    state.board[4][1] = Some(Player::Yellow);
    state.board[3][1] = Some(Player::Red);

    // Col 2: 1 dummy piece, then Red
    state.board[5][2] = Some(Player::Yellow);
    state.board[4][2] = Some(Player::Red);

    // Col 3: Red at bottom
    state.board[5][3] = Some(Player::Red);

    assert!(state.check_win_at(5, 3, Player::Red));
    assert!(state.has_won(Player::Red));
    assert!(!state.has_won(Player::Yellow));
}

#[test]
fn test_diagonal_up_right_win() {
    let mut state = Connect4State::<6, 7>::new();
    // Diagonal up-right: (5, 0), (4, 1), (3, 2), (2, 3)
    state.board[5][0] = Some(Player::Red);

    state.board[5][1] = Some(Player::Yellow);
    state.board[4][1] = Some(Player::Red);

    state.board[5][2] = Some(Player::Yellow);
    state.board[4][2] = Some(Player::Yellow);
    state.board[3][2] = Some(Player::Red);

    state.board[5][3] = Some(Player::Yellow);
    state.board[4][3] = Some(Player::Yellow);
    state.board[3][3] = Some(Player::Yellow);
    state.board[2][3] = Some(Player::Red);

    assert!(state.check_win_at(2, 3, Player::Red));
    assert!(state.has_won(Player::Red));
}

#[test]
fn test_dynamics_step_and_turn_alternation() {
    let env = Connect4Dynamics::<6, 7>;
    let mut state = AgentDynamics::initial(&env);

    assert_eq!(state.current_player, Player::Red);
    let outcome = AgentDynamics::step(&env, &mut state, &3);
    assert!(!outcome.terminated);
    assert_eq!(outcome.reward, [0.0, 0.0]);
    assert_eq!(state.current_player, Player::Yellow);

    let outcome2 = AgentDynamics::step(&env, &mut state, &3);
    assert!(!outcome2.terminated);
    assert_eq!(state.current_player, Player::Red);
}

#[test]
fn test_world_referee_actions_and_terminal() {
    let env = Connect4Dynamics::<6, 7>;
    let mut state = World::initial(&env);

    let mut actions_p0 = Vec::new();
    let mut actions_p1 = Vec::new();

    // Player 0 is active
    World::actions(&env, &state, 0, &mut actions_p0);
    World::actions(&env, &state, 1, &mut actions_p1);
    assert_eq!(actions_p0.len(), 7);
    assert!(actions_p1.is_empty());

    // Step with joint action: player 0 plays col 0
    let (rewards, term) = World::step(&env, &mut state, &[0, 0]);
    assert!(!term);
    assert_eq!(rewards, vec![0.0, 0.0]);

    // Now player 1 is active
    World::actions(&env, &state, 0, &mut actions_p0);
    World::actions(&env, &state, 1, &mut actions_p1);
    assert!(actions_p0.is_empty());
    assert_eq!(actions_p1.len(), 7);
}

#[test]
fn test_evaluators() {
    let state = Connect4State::<6, 7>::new();

    let uniform = UniformEvaluator;
    let eval_u = uniform.evaluate(&state);
    assert_eq!(eval_u.priors.len(), 7);
    assert!((eval_u.priors[0] - 1.0 / 7.0).abs() < 1e-5);
    assert_eq!(eval_u.values, vec![0.0, 0.0]);

    let rollout = RolloutEvaluator::<6, 7>::new(10, 15);
    let eval_r = rollout.evaluate(&state);
    assert_eq!(eval_r.priors.len(), 7);
    assert_eq!(eval_r.values.len(), 2);
}

#[test]
fn test_random_agent_legal_moves() {
    let mut agent = RandomAgent::new("Bot");
    assert_eq!(agent.name(), "Bot");

    let state = Connect4State::<6, 7>::new();
    for _ in 0..20 {
        let act = agent.select_action(&state);
        assert!(act < 7);
    }
}

#[test]
fn test_mcts_agent_finds_immediate_winning_move() {
    // Set up a board where Red has 3 in a row at bottom (cols 0, 1, 2)
    // Red must pick col 3 to win immediately!
    let mut state = Connect4State::<6, 7>::new();
    state.board[5][0] = Some(Player::Red);
    state.board[5][1] = Some(Player::Red);
    state.board[5][2] = Some(Player::Red);
    state.current_player = Player::Red;

    let mut agent = MctsAgent::new_uniform("MCTS-Red", 80, false);
    let chosen = agent.select_action(&state);
    assert_eq!(chosen, 3, "MCTS should pick winning col 3");
}

#[test]
fn test_render_board_output() {
    let state = Connect4State::<6, 7>::new();
    let rendered = render_board(&state);
    assert!(rendered.contains("0    1    2    3    4    5    6") || rendered.contains("0"));
    assert!(rendered.contains("┌") && rendered.contains("┘"));

    let styled = render_board_styled(&state, true);
    assert!(styled.contains("\x1b[34m"));
}

#[test]
fn test_mcts_agent_responds_to_center_opening() {
    let mut state = Connect4State::<6, 7>::new();
    state.drop_piece(3).unwrap();
    state.current_player = Player::Yellow;

    let mut agent = MctsAgent::new_rollout("Yellow", 2000, 5, 20, false);
    let chosen = agent.select_action(&state);
    // Yellow should contest the central columns (2, 3, or 4), not the extreme flank (col 0 or 6)
    assert!(
        chosen == 2 || chosen == 3 || chosen == 4,
        "Yellow should contest central columns (2, 3, or 4), but chose col {chosen}"
    );
}

#[test]
fn test_mcts_agent_blocks_open_ended_threat() {
    let mut state = Connect4State::<6, 7>::new();
    state.board[5][6] = Some(Player::Yellow);
    state.board[4][6] = Some(Player::Yellow);
    state.board[3][6] = Some(Player::Yellow);
    state.board[2][6] = Some(Player::Red);

    state.board[5][3] = Some(Player::Red);
    state.board[4][3] = Some(Player::Red);
    state.board[3][3] = Some(Player::Red);
    state.board[2][3] = Some(Player::Yellow);

    state.board[5][2] = Some(Player::Red);

    state.current_player = Player::Yellow;

    let mut agent = MctsAgent::new_rollout("Yellow", 3000, 5, 20, false);
    let chosen = agent.select_action(&state);
    // Yellow must block either col 1 or col 4 to prevent Red forming an open-ended 3
    assert!(
        chosen == 1 || chosen == 4,
        "Yellow should play col 1 or 4 to block Red's threat, but chose col {chosen}"
    );
}




