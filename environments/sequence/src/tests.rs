//! Comprehensive unit tests for Sequence game rules, sequence detection, determinization, and MCTS agents.

use crate::board::{
    card_positions, coord_to_index, create_double_deck, detect_sequences, index_to_coord, is_corner,
    is_corner_index, is_jack, is_one_eyed_jack, is_two_eyed_jack, BoardCell, SequenceRecord,
    BOARD_CELLS, BOARD_LAYOUT,
};
use crate::dynamics::determinize_state;
use crate::game::{SequenceAction, SequenceConfig, SequenceState};
use crate::world::{Sequence2PWorld, Sequence3PWorld};
use cardpack::prelude::FrenchBasicCard;
use mcts_engine::arena::MatchDriver;
use mcts_traits::{Agent, World};

#[test]
fn test_double_deck_composition() {
    let deck = create_double_deck();
    assert_eq!(deck.len(), 104, "Double deck must contain exactly 104 cards");

    let mut jacks = 0;
    let mut two_eyed = 0;
    let mut one_eyed = 0;

    for &c in &deck {
        if is_jack(c) {
            jacks += 1;
        }
        if is_two_eyed_jack(c) {
            two_eyed += 1;
        }
        if is_one_eyed_jack(c) {
            one_eyed += 1;
        }
    }

    assert_eq!(jacks, 8, "Double deck must have 8 Jacks");
    assert_eq!(two_eyed, 4, "Double deck must have 4 Two-Eyed Jacks (2x JC, 2x JD)");
    assert_eq!(one_eyed, 4, "Double deck must have 4 One-Eyed Jacks (2x JS, 2x JH)");
}

#[test]
fn test_board_layout_card_counts_and_corners() {
    let mut corner_count = 0;
    let mut card_counts = std::collections::HashMap::new();

    for (idx, &cell) in BOARD_LAYOUT.iter().enumerate() {
        let (r, c) = index_to_coord(idx);
        assert_eq!(coord_to_index(r, c), idx, "Coordinate conversion must roundtrip");

        match cell {
            BoardCell::Corner => {
                corner_count += 1;
                assert!(is_corner(r, c), "Corner cell must be at board corner");
                assert!(is_corner_index(idx), "Corner index check must match");
            }
            BoardCell::Card(card) => {
                assert!(!is_jack(card), "Jacks must not appear on board spaces");
                *card_counts.entry(card).or_insert(0) += 1;
            }
        }
    }

    assert_eq!(corner_count, 4, "Board must have exactly 4 corner wild spaces");
    assert_eq!(card_counts.len(), 48, "Board must contain 48 distinct non-Jack cards");
    for (card, count) in card_counts {
        assert_eq!(count, 2, "Card {card} must appear exactly twice on the board");
        let positions = card_positions(card);
        assert_ne!(positions[0], positions[1], "Two card positions must be distinct");
    }
}

#[test]
fn test_sequence_detection_horizontal_and_localized_equivalence() {
    let mut board: [Option<u8>; BOARD_CELLS] = [None; BOARD_CELLS];
    let mut locked_chips = [false; BOARD_CELLS];
    let mut seqs = Vec::new();

    // Place 5 chips horizontally for team 0 at row 1, cols 1..=5
    for c in 1..=5 {
        board[coord_to_index(1, c)] = Some(0);
    }

    // Localized check at last placed (1, 5)
    let detected_localized = detect_sequences(
        &board,
        0,
        Some((1, 5)),
        &mut locked_chips,
        &mut seqs,
    );
    assert_eq!(detected_localized, 1, "Should detect 1 horizontal sequence");
    assert_eq!(seqs.len(), 1);

    // Verify chips are locked
    for c in 1..=5 {
        assert!(locked_chips[coord_to_index(1, c)], "Sequence chips must be locked");
    }

    // Now test full board scan equivalence
    let mut locked_full = [false; BOARD_CELLS];
    let mut seqs_full = Vec::new();
    let detected_full = detect_sequences(&board, 0, None, &mut locked_full, &mut seqs_full);
    assert_eq!(detected_full, 1);
    assert_eq!(locked_chips, locked_full);
    assert_eq!(seqs, seqs_full);
}

#[test]
fn test_sequence_detection_corner_sharing() {
    let mut board: [Option<u8>; BOARD_CELLS] = [None; BOARD_CELLS];
    let mut locked = [false; BOARD_CELLS];
    let mut seqs = Vec::new();

    // Place 4 chips horizontally in row 0 at cols 1..4 (using corner (0, 0) as 5th space)
    for c in 1..=4 {
        board[coord_to_index(0, c)] = Some(1);
    }

    let detected = detect_sequences(&board, 1, Some((0, 4)), &mut locked, &mut seqs);
    assert_eq!(detected, 1, "Sequence should form using corner (0, 0)");
    assert!(!locked[0], "Corner spaces are never locked to one team");
    for c in 1..=4 {
        assert!(locked[coord_to_index(0, c)], "Non-corner chips must be locked");
    }
}

#[test]
fn test_sequence_detection_nine_in_a_row_double_sequence() {
    let mut board: [Option<u8>; BOARD_CELLS] = [None; BOARD_CELLS];
    let mut locked = [false; BOARD_CELLS];
    let mut seqs = Vec::new();

    // Place 9 chips in row 2 at cols 1..=9
    for c in 1..=8 {
        board[coord_to_index(2, c)] = Some(0);
        detect_sequences(&board, 0, Some((2, c)), &mut locked, &mut seqs);
    }
    assert_eq!(seqs.len(), 1, "First sequence forms at 5 chips");

    // 9th chip placed at (2, 9)
    board[coord_to_index(2, 9)] = Some(0);
    let detected = detect_sequences(&board, 0, Some((2, 9)), &mut locked, &mut seqs);
    assert_eq!(detected, 1, "9-in-a-row should form second sequence sharing chip 5");
    assert_eq!(seqs.len(), 2);
}

#[test]
fn test_sequence_detection_intersection_limit() {
    let mut board: [Option<u8>; BOARD_CELLS] = [None; BOARD_CELLS];
    let mut locked = [false; BOARD_CELLS];
    let mut seqs = Vec::new();

    // Sequence 1: horizontal row 4, cols 1..=5
    for c in 1..=5 {
        board[coord_to_index(4, c)] = Some(0);
    }
    assert_eq!(detect_sequences(&board, 0, Some((4, 5)), &mut locked, &mut seqs), 1);

    // Sequence 2: vertical col 3, rows 2..=6 (intersects Sequence 1 at (4, 3) - 1 non-corner chip)
    for r in 2..=6 {
        board[coord_to_index(r, 3)] = Some(0);
    }
    assert_eq!(detect_sequences(&board, 0, Some((6, 3)), &mut locked, &mut seqs), 1);
    assert_eq!(seqs.len(), 2, "2 sequences sharing 1 non-corner chip are valid");

    // Sequence 3 candidate: diagonal passing through BOTH (4, 2) and (4, 3) would share 2 chips -> illegal!
    let cand = SequenceRecord::new(0, [
        coord_to_index(4, 2) as u8,
        coord_to_index(4, 3) as u8,
        coord_to_index(5, 4) as u8,
        coord_to_index(6, 5) as u8,
        coord_to_index(7, 6) as u8,
    ]);
    assert_eq!(cand.shared_non_corner_count(&seqs[0]), 2);
}

#[test]
fn test_two_eyed_jack_and_one_eyed_jack_mechanics() {
    let config = SequenceConfig::new_2p();
    let mut rng = rand::thread_rng();
    let mut state = SequenceState::new(config, &mut rng);

    // Give player 0 a Two-Eyed Jack (Wild) and player 1 a One-Eyed Jack (Removal)
    let wild_card = FrenchBasicCard::JACK_CLUBS;
    let removal_card = FrenchBasicCard::JACK_SPADES;

    state.hands[0] = vec![wild_card];
    state.hands[1] = vec![removal_card];

    // Player 0 plays Two-Eyed Jack on (5, 5)
    let play_wild = SequenceAction::PlayCard {
        card: wild_card,
        pos: (5, 5),
    };
    state.step(&play_wild, &mut rng);

    let idx = coord_to_index(5, 5);
    assert_eq!(state.board[idx], Some(0), "Wild Jack must place team 0 token at (5, 5)");
    assert_eq!(state.current_player, 1, "Turn should advance to Player 1");

    // Player 1 uses One-Eyed Jack to remove player 0's token at (5, 5)
    let mut legal_p1 = Vec::new();
    state.legal_actions(&mut legal_p1);
    let remove_action = SequenceAction::RemoveToken {
        card: removal_card,
        pos: (5, 5),
    };
    assert!(legal_p1.contains(&remove_action), "One-Eyed Jack should be able to remove opponent token");

    state.step(&remove_action, &mut rng);
    assert_eq!(state.board[idx], None, "Removed token space must now be empty");
}

#[test]
fn test_locked_chip_immunity_to_one_eyed_jack() {
    let config = SequenceConfig::new_2p();
    let mut rng = rand::thread_rng();
    let mut state = SequenceState::new(config, &mut rng);

    // Manually complete a sequence for team 0
    for c in 1..=5 {
        let idx = coord_to_index(3, c);
        state.board[idx] = Some(0);
        state.locked_chips[idx] = true;
    }
    state.sequences.push(SequenceRecord::new(0, [
        coord_to_index(3, 1) as u8,
        coord_to_index(3, 2) as u8,
        coord_to_index(3, 3) as u8,
        coord_to_index(3, 4) as u8,
        coord_to_index(3, 5) as u8,
    ]));
    state.team_sequence_counts[0] = 1;

    // Player 1 has One-Eyed Jack
    state.current_player = 1;
    state.hands[1] = vec![FrenchBasicCard::JACK_SPADES];

    let mut legal_p1 = Vec::new();
    state.legal_actions(&mut legal_p1);

    // Verify player 1 cannot remove any locked chip
    for c in 1..=5 {
        let forbidden = SequenceAction::RemoveToken {
            card: FrenchBasicCard::JACK_SPADES,
            pos: (3, c),
        };
        assert!(!legal_p1.contains(&forbidden), "One-Eyed Jack cannot target locked chips");
    }
}

#[test]
fn test_dead_card_exchange() {
    let config = SequenceConfig::new_2p();
    let mut rng = rand::thread_rng();
    let mut state = SequenceState::new(config, &mut rng);

    let card = FrenchBasicCard::DEUCE_SPADES;
    let positions = card_positions(card);

    // Occupy both board spaces for DEUCE_SPADES
    state.board[coord_to_index(positions[0].0, positions[0].1)] = Some(1);
    state.board[coord_to_index(positions[1].0, positions[1].1)] = Some(0);

    assert!(state.is_dead_card(card), "Card should be dead when both board spots are occupied");

    state.hands[0] = vec![card];
    let mut legal = Vec::new();
    state.legal_actions(&mut legal);

    assert_eq!(
        legal,
        vec![SequenceAction::DiscardDeadCard { card }],
        "Only legal action for dead card is DiscardDeadCard"
    );

    let initial_hand_len = state.hands[0].len();
    state.step(&legal[0], &mut rng);

    assert_eq!(state.hands[0].len(), initial_hand_len, "Player draws a replacement for discarded dead card");
    assert_eq!(state.discards.last(), Some(&card), "Dead card must be moved to discard pile");
}

#[test]
fn test_determinization_card_conservation() {
    let world = Sequence2PWorld::default();
    let initial = world.initial();
    let obs = world.observe(&initial, 0);

    assert_eq!(obs.my_hand.len(), 7);
    assert_eq!(obs.other_hand_counts, vec![7, 7]);
    assert_eq!(obs.deck_remaining, 104 - 14);

    let mut rng = rand::thread_rng();
    let det = determinize_state(&obs, &mut rng);

    // Exact card conservation checks
    assert_eq!(det.hands[0], obs.my_hand, "Observer hand must match exactly");
    assert_eq!(det.hands[1].len(), 7, "Opponent hand must have 7 cards");
    assert_eq!(det.deck.len(), 104 - 14, "Remaining deck count must be preserved");

    // Total cards across all hands + deck + discards must equal 104
    let total_cards: usize = det.hands.iter().map(|h| h.len()).sum::<usize>() + det.deck.len() + det.discards.len();
    assert_eq!(total_cards, 104, "Total cards in determinized state must equal 104");
}

#[test]
fn test_world_3p_zero_sum_rewards() {
    let world = Sequence3PWorld::default();
    let mut ws = world.initial();
    assert_eq!(world.n_players(), 3);
    assert_eq!(ws.config.num_teams, 3);
    assert_eq!(ws.config.target_sequences, 1);

    // Simulate team 1 winning
    ws.terminated = true;
    ws.winner_team = Some(1);

    let rewards = world.compute_rewards(&ws);
    assert_eq!(rewards, [-0.5, 1.0, -0.5]);
    let sum: f32 = rewards.iter().sum();
    assert!((sum - 0.0).abs() < 1e-6, "3-player rewards must sum to 0.0");
}

#[test]
fn test_tournament_short_match_between_heuristic_and_random() {
    let world = Sequence2PWorld::default();
    let driver = MatchDriver::new();

    let mut a0 = crate::agent::HeuristicAgent::new("HeuristicBot");
    let mut a1 = crate::agent::RandomAgent::new("RandomBot");

    let result = driver.play_2p(&world, &mut a0, &mut a1, None);
    assert!(result.total_moves > 0);
    assert!(result.final_state.terminated);
}

#[test]
fn test_ismcts_and_opponent_model_mcts_step() {
    let world = Sequence2PWorld::default();
    let state = world.initial();

    let mut ismcts = crate::agent::IsMctsAgent::<crate::evaluator::SequenceHeuristicEvaluator, 2>::new_heuristic("ISMCTS", 20, 2);
    let mut opp_mcts = crate::agent::OpponentModelMctsAgent::<_, 2>::new_heuristic("OppModelMCTS", 20, 2);

    let action_ismcts = ismcts.select_action(&state);
    let action_opp = opp_mcts.select_action(&state);

    let mut legal = Vec::new();
    state.legal_actions(&mut legal);
    assert!(legal.contains(&action_ismcts), "ISMCTS chosen action must be legal");
    assert!(legal.contains(&action_opp), "OpponentModel MCTS chosen action must be legal");
}
