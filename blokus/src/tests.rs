//! Unit tests for Blokus pieces, game rules, scoring, and multi-agent MCTS planning.

use crate::agent::{Agent, HeuristicAgent, MctsAgent, RandomAgent};
use crate::dynamics::{compute_rank_rewards, BlokusDuoDynamics};
use crate::game::{BlokusAction, BlokusClassicState, BlokusDuoState};
use crate::pieces::{piece_size, registry, NUM_PIECES, TOTAL_SQUARES_PER_PLAYER};
use mcts_traits::World;

#[test]
fn test_piece_registry_and_canonical_orientations() {
    let reg = registry();
    let mut total_orientations = 0;
    let mut total_squares = 0;

    for i in 0..NUM_PIECES {
        let size = piece_size(i as u8);
        total_squares += size as usize;
        let orientations = reg.orientations_of(i);
        assert!(!orientations.is_empty(), "Piece {i} must have orientations");
        assert!(orientations.len() <= 8, "Orientation count must not exceed 8");

        for shape in orientations {
            assert_eq!(shape.num_squares, size);
            assert!(shape.height >= 1 && shape.height <= 5);
            assert!(shape.width >= 1 && shape.width <= 5);
            // Verify normalization: min r == 0 and min c == 0
            let min_r = shape.active_squares().iter().map(|&(r, _)| r).min().unwrap();
            let min_c = shape.active_squares().iter().map(|&(_, c)| c).min().unwrap();
            assert_eq!(min_r, 0);
            assert_eq!(min_c, 0);
        }
        total_orientations += orientations.len();
    }

    assert_eq!(total_squares, TOTAL_SQUARES_PER_PLAYER);
    assert_eq!(total_orientations, 91, "Expected exactly 91 canonical polyomino orientations");
}

#[test]
fn test_start_square_rule_classic_and_duo() {
    let mut classic = BlokusClassicState::new();
    assert_eq!(classic.start_square(0), (0, 0));
    assert_eq!(classic.start_square(1), (0, 19));
    assert_eq!(classic.start_square(2), (19, 19));
    assert_eq!(classic.start_square(3), (19, 0));

    // Player 0 tries to place monomino (piece 0) not at (0, 0)
    let invalid_first_move = BlokusAction::Place {
        piece_id: 0,
        orientation: 0,
        row: 5,
        col: 5,
    };
    assert!(classic.apply_action(&invalid_first_move).is_err());

    // Valid first move covering (0, 0)
    let valid_first_move = BlokusAction::Place {
        piece_id: 0,
        orientation: 0,
        row: 0,
        col: 0,
    };
    assert!(classic.apply_action(&valid_first_move).is_ok());
    assert_eq!(classic.board[0][0], 0);

    // Blokus Duo starts at (4, 4) and (9, 9)
    let duo = BlokusDuoState::new();
    assert_eq!(duo.start_square(0), (4, 4));
    assert_eq!(duo.start_square(1), (9, 9));
}

#[test]
fn test_blokus_adjacency_rules() {
    let mut state = BlokusDuoState::new();
    // Player 0 places Domino (piece 1, horizontal: (0,0), (0,1)) covering (4, 4) and (4, 5)
    let m0 = BlokusAction::Place {
        piece_id: 1,
        orientation: 0,
        row: 4,
        col: 4,
    };
    assert!(state.apply_action(&m0).is_ok());

    // Player 1 places Domino covering (9, 9) and (9, 10)
    let m1 = BlokusAction::Place {
        piece_id: 1,
        orientation: 0,
        row: 9,
        col: 9,
    };
    assert!(state.apply_action(&m1).is_ok());

    // Now it's Player 0's second turn.
    // Placing monomino at (4, 3) shares an edge with (4, 4) -> ILLEGAL!
    let illegal_edge_touch = BlokusAction::Place {
        piece_id: 0,
        orientation: 0,
        row: 4,
        col: 3,
    };
    assert!(state.apply_action(&illegal_edge_touch).is_err());

    // Placing monomino at (3, 4) shares an edge with (4, 4) -> ILLEGAL!
    let illegal_edge_touch_top = BlokusAction::Place {
        piece_id: 0,
        orientation: 0,
        row: 3,
        col: 4,
    };
    assert!(state.apply_action(&illegal_edge_touch_top).is_err());

    // Placing monomino at (3, 3) touches (4, 4) diagonally at corner -> LEGAL!
    let legal_corner_touch = BlokusAction::Place {
        piece_id: 0,
        orientation: 0,
        row: 3,
        col: 3,
    };
    assert!(state.apply_action(&legal_corner_touch).is_ok());
    assert_eq!(state.board[3][3], 0);
}

#[test]
fn test_legal_actions_generation_and_free_corners() {
    let mut state = BlokusClassicState::new();
    let mut actions = Vec::new();
    state.legal_actions(&mut actions);

    // Initial moves must all cover (0, 0)
    assert!(!actions.is_empty());
    for act in &actions {
        match act {
            BlokusAction::Place {
                piece_id,
                orientation,
                row,
                col,
            } => {
                assert!(state.is_valid_placement(0, *piece_id, *orientation, *row, *col));
            }
            BlokusAction::Pass => panic!("Pass should not be legal on first turn"),
        }
    }

    // Play first move: monomino at (0, 0)
    state
        .apply_action(&BlokusAction::Place {
            piece_id: 0,
            orientation: 0,
            row: 0,
            col: 0,
        })
        .unwrap();

    let mut corners = Vec::new();
    state.find_valid_corners(0, &mut corners);
    // (0, 0) has diagonal (1, 1). Orthogonal neighbors (0, 1) and (1, 0) are blocked by edge rule.
    assert_eq!(corners, vec![(1, 1)]);
}

#[test]
fn test_pass_mechanics_and_skip_turn() {
    let mut state = BlokusClassicState::new();
    assert_eq!(state.current_player, 0);

    // Player 0 passes
    state.apply_action(&BlokusAction::Pass).unwrap();
    assert_eq!(state.current_player, 1);
    assert!(state.passed[0]);

    // Player 1 passes
    state.apply_action(&BlokusAction::Pass).unwrap();
    assert_eq!(state.current_player, 2);
    assert!(state.passed[1]);

    // Player 2 passes
    state.apply_action(&BlokusAction::Pass).unwrap();
    assert_eq!(state.current_player, 3);
    assert!(state.passed[2]);

    // Player 3 passes -> all players passed, game is terminal
    state.apply_action(&BlokusAction::Pass).unwrap();
    assert!(state.is_terminal());
}

#[test]
fn test_scoring_rules_and_bonuses() {
    let mut state = BlokusClassicState::new();
    // Initially unplaced squares = 89, score = -89
    assert_eq!(state.unplaced_squares(0), 89);
    assert_eq!(state.score(0), -89);

    // Simulate placing all pieces except monomino
    state.remaining_pieces[0] = 1 << 0; // only monomino left
    assert_eq!(state.unplaced_squares(0), 1);
    assert_eq!(state.score(0), -1);

    // Place monomino as last piece
    state.remaining_pieces[0] = 0;
    state.last_piece[0] = Some(0);
    assert_eq!(state.score(0), 20); // +20 bonus for monomino last!

    // If last piece was piece 1 (domino)
    state.last_piece[0] = Some(1);
    assert_eq!(state.score(0), 15); // +15 bonus for any other piece last
}

#[test]
fn test_compute_rank_rewards_zero_sum() {
    // 2 players distinct
    let r2 = compute_rank_rewards(&[10, -5]);
    assert_eq!(r2, [1.0, -1.0]);
    assert_eq!(r2[0] + r2[1], 0.0);

    // 2 players tie
    let r2_tie = compute_rank_rewards(&[5, 5]);
    assert_eq!(r2_tie, [0.0, 0.0]);

    // 4 players distinct
    let r4 = compute_rank_rewards(&[20, 10, -5, -20]);
    assert!((r4.iter().sum::<f32>()).abs() < 1e-5);
    assert_eq!(r4[0], 1.0);
    assert_eq!(r4[3], -1.0);

    // 4 players with 2-way tie
    let r4_tie = compute_rank_rewards(&[10, 10, 0, -10]);
    assert!((r4_tie.iter().sum::<f32>()).abs() < 1e-5);
}

#[test]
fn test_dynamics_and_world_trait() {
    let env = BlokusDuoDynamics::default();
    let mut ws = World::initial(&env);
    assert_eq!(env.n_players(), 2);
    assert!(!env.terminal(&ws));

    let mut actions = Vec::new();
    World::actions(&env, &ws, 0, &mut actions);
    assert!(!actions.is_empty());

    let (rewards, term) = World::step(&env, &mut ws, &[actions[0], BlokusAction::Pass]);
    assert_eq!(rewards.len(), 2);
    assert!(!term);
}

#[test]
fn test_heuristic_and_random_agent_match() {
    let mut state = BlokusDuoState::new();
    let mut p0 = HeuristicAgent::new("Heuristic-0");
    let mut p1 = RandomAgent::new("Random-1");

    let mut moves = 0;
    while !state.is_terminal() && moves < 20 {
        let act = if state.current_player == 0 {
            p0.select_action(&state)
        } else {
            p1.select_action(&state)
        };
        state.apply_action(&act).unwrap();
        moves += 1;
    }

    assert!(moves > 0);
    assert!(state.score(0) > -89);
}

#[test]
fn test_mcts_agent_search_2p_and_4p() {
    // 2-player Blokus Duo MCTS
    let duo_state = BlokusDuoState::new();
    let mut duo_mcts = MctsAgent::new_heuristic("MCTS-Duo", 25, false);
    let duo_move = duo_mcts.select_action(&duo_state);
    assert!(matches!(duo_move, BlokusAction::Place { .. }));

    // 4-player Blokus Classic MCTS
    let classic_state = BlokusClassicState::new();
    let mut classic_mcts = MctsAgent::new_heuristic("MCTS-Classic", 25, false);
    let classic_move = classic_mcts.select_action(&classic_state);
    assert!(matches!(classic_move, BlokusAction::Place { .. }));
}
