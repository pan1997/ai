//! Unit tests for Hex game logic, DSU win tracking, evaluators, and search agents.

use crate::agent::{Agent, HeuristicAgent, HexAgentSpec, MctsAgent};
use crate::evaluator::ShortestPathHeuristicEvaluator;
use crate::game::{HexPlayer, HexState};
use crate::world::HexWorld;
use mcts_traits::{TurnBasedWorld, World};

#[test]
fn test_coordinate_roundtrip() {
    assert_eq!(HexState::<11>::coord_to_str(0), "A1");
    assert_eq!(HexState::<11>::coord_to_str(10), "K1");
    assert_eq!(HexState::<11>::coord_to_str(11), "A2");
    assert_eq!(HexState::<11>::coord_to_str(120), "K11");

    assert_eq!(HexState::<11>::str_to_coord("A1"), Some(0));
    assert_eq!(HexState::<11>::str_to_coord("a1"), Some(0));
    assert_eq!(HexState::<11>::str_to_coord("K1"), Some(10));
    assert_eq!(HexState::<11>::str_to_coord("A2"), Some(11));
    assert_eq!(HexState::<11>::str_to_coord("K11"), Some(120));
    assert_eq!(HexState::<11>::str_to_coord("0 0"), Some(0));
    assert_eq!(HexState::<11>::str_to_coord("10 10"), Some(120));
    assert_eq!(HexState::<11>::str_to_coord("120"), Some(120));
    assert_eq!(HexState::<11>::str_to_coord("Z99"), None);
}

#[test]
fn test_black_win_connection() {
    let mut state = HexState::<3>::new();
    // Black connects Top to Bottom along column 0: (0,0), (1,0), (2,0)
    assert!(!state.play_move(0)); // Black at (0,0)
    assert_eq!(state.current_player, HexPlayer::Black);
    state.current_player = HexPlayer::White;
    assert!(!state.play_move(1)); // White at (0,1)
    state.current_player = HexPlayer::Black;
    assert!(!state.play_move(3)); // Black at (1,0)
    state.current_player = HexPlayer::White;
    assert!(!state.play_move(4)); // White at (1,1)
    state.current_player = HexPlayer::Black;
    let won = state.play_move(6); // Black at (2,0)
    assert!(won, "Black should have won connecting (0,0)-(1,0)-(2,0)");
    assert!(state.is_won(HexPlayer::Black));
    assert!(!state.is_won(HexPlayer::White));
}

#[test]
fn test_white_win_connection() {
    let mut state = HexState::<3>::new();
    // White connects Left to Right along row 0: (0,0), (0,1), (0,2)
    state.current_player = HexPlayer::White;
    assert!(!state.play_move(0)); // White at (0,0)
    state.current_player = HexPlayer::Black;
    assert!(!state.play_move(3)); // Black at (1,0)
    state.current_player = HexPlayer::White;
    assert!(!state.play_move(1)); // White at (0,1)
    state.current_player = HexPlayer::Black;
    assert!(!state.play_move(4)); // Black at (1,1)
    state.current_player = HexPlayer::White;
    let won = state.play_move(2); // White at (0,2)
    assert!(won, "White should have won connecting (0,0)-(0,1)-(0,2)");
    assert!(state.is_won(HexPlayer::White));
    assert!(!state.is_won(HexPlayer::Black));
}

#[test]
fn test_hex_world_referee() {
    let world = HexWorld::<3>::new();
    let mut state = world.initial();
    assert_eq!(world.n_players(), 2);
    assert_eq!(world.current_player(&state), 0); // Black

    let mut actions = Vec::new();
    world.actions(&state, 0, &mut actions);
    assert_eq!(actions.len(), 9);

    let outcome = world.step_action(&mut state, 4); // Black plays center (1,1)
    assert!(!outcome.terminated);
    assert_eq!(outcome.reward, [0.0, 0.0]);
    assert_eq!(state.current_player, HexPlayer::White);
    assert_eq!(world.current_player(&state), 1);
}

#[test]
fn test_shortest_path_heuristic() {
    let mut state = HexState::<5>::new();
    // Empty board distance
    let d_black = ShortestPathHeuristicEvaluator::<5>::black_shortest_path(&state);
    let d_white = ShortestPathHeuristicEvaluator::<5>::white_shortest_path(&state);
    assert_eq!(d_black, 5);
    assert_eq!(d_white, 5);

    // Initial symmetry: advantage should be 0.0
    let eval = ShortestPathHeuristicEvaluator::<5>::evaluate_state(&state);
    assert_eq!(eval, 0.0);

    // Play Black stone at center (2,2)
    state.play_move(HexState::<5>::idx(2, 2));
    let d_black_after = ShortestPathHeuristicEvaluator::<5>::black_shortest_path(&state);
    assert_eq!(d_black_after, 4);
    let eval_after = ShortestPathHeuristicEvaluator::<5>::evaluate_state(&state);
    assert!(
        eval_after > 0.0,
        "Black should have positive advantage after center play"
    );
}

#[test]
fn test_heuristic_agent_takes_win() {
    let mut state = HexState::<3>::new();
    // Set up Black 1 move away from winning: (0,0) and (1,0) placed
    state.play_move(0); // Black (0,0)
    state.current_player = HexPlayer::White;
    state.play_move(1); // White (0,1)
    state.current_player = HexPlayer::Black;
    state.play_move(3); // Black (1,0)
    state.current_player = HexPlayer::White;
    state.play_move(4); // White (1,1)
    state.current_player = HexPlayer::Black;

    let mut agent = HeuristicAgent::<3>::new();
    let action = agent.select_action(&state);
    assert_eq!(
        action, 6,
        "HeuristicAgent should immediately choose winning move (2,0) = 6"
    );
}

#[test]
fn test_mcts_agent_takes_immediate_win() {
    let mut state = HexState::<3>::new();
    // Set up Black 1 move away from winning: (0,0) and (1,0) placed
    state.play_move(0); // Black (0,0)
    state.current_player = HexPlayer::White;
    state.play_move(1); // White (0,1)
    state.current_player = HexPlayer::Black;
    state.play_move(3); // Black (1,0)
    state.current_player = HexPlayer::White;
    state.play_move(4); // White (1,1)
    state.current_player = HexPlayer::Black;

    let mut agent = MctsAgent::<3>::new_heuristic("Mcts-H", 200, false);
    let action = agent.select_action(&state);
    assert_eq!(
        action, 6,
        "MctsAgent should choose winning move (2,0) = 6 to complete path"
    );
}

#[test]
fn test_spec_parser() {
    let s1 = HexAgentSpec::parse("mcts:1000:3:1.5").unwrap();
    assert_eq!(
        s1,
        HexAgentSpec::MctsRollout {
            iters: 1000,
            rollouts: 3,
            c_puct: 1.5,
        }
    );

    let s2 = HexAgentSpec::parse("mcts-h:300").unwrap();
    assert_eq!(
        s2,
        HexAgentSpec::MctsHeuristic {
            iters: 300,
            c_puct: 1.414,
        }
    );

    let s3 = HexAgentSpec::parse("mcts-u:200").unwrap();
    assert_eq!(
        s3,
        HexAgentSpec::MctsUniform {
            iters: 200,
            c_puct: 1.414,
        }
    );

    let s4 = HexAgentSpec::parse("heuristic").unwrap();
    assert_eq!(s4, HexAgentSpec::Heuristic);

    let s5 = HexAgentSpec::parse("random").unwrap();
    assert_eq!(s5, HexAgentSpec::Random);

    let s6 = HexAgentSpec::parse("human").unwrap();
    assert_eq!(s6, HexAgentSpec::Human);

    assert!(HexAgentSpec::parse("unknown_agent").is_err());
}

#[test]
fn test_batch_vs_sequential_rollout_equivalence() {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let mut state = HexState::<5>::new();
        // Play a random number of opening moves (0 to 10)
        let num_opening = rand::Rng::gen_range(&mut rng, 0..10);
        for _ in 0..num_opening {
            let mut legals = Vec::new();
            state.legal_actions(&mut legals);
            if legals.is_empty() || state.is_won(HexPlayer::Black) || state.is_won(HexPlayer::White)
            {
                break;
            }
            let choice = *legals.choose(&mut rng).unwrap();
            state.play_move(choice);
            state.current_player = state.current_player.other();
        }

        if state.is_won(HexPlayer::Black) || state.is_won(HexPlayer::White) {
            continue;
        }

        let mut empty = Vec::new();
        for idx in 0..25 {
            if state.board[idx].is_none() {
                empty.push(idx);
            }
        }
        empty.shuffle(&mut rng);

        // 1. Sequential rollout
        let mut seq_state = state.clone();
        let mut seq_winner = None;
        for &cell in &empty {
            let p = seq_state.current_player;
            let won = seq_state.play_move(cell);
            if won {
                seq_winner = Some(p);
                break;
            }
            seq_state.current_player = p.other();
        }

        // 2. Batch rollout (Black-only DSU updates!)
        let mut batch_board = state.board;
        let mut batch_dsu_black = state.dsu_black;
        let mut curr = state.current_player;
        for &cell in &empty {
            batch_board[cell] = Some(curr);
            if curr == HexPlayer::Black {
                let (r, c) = HexState::<5>::coord(cell);
                if r == 0 {
                    HexState::<5>::dsu_union(&mut batch_dsu_black, cell, 25);
                }
                if r == 4 {
                    HexState::<5>::dsu_union(&mut batch_dsu_black, cell, 26);
                }
                for (nr, nc) in HexState::<5>::neighbors(r, c) {
                    let n_idx = HexState::<5>::idx(nr, nc);
                    if batch_board[n_idx] == Some(HexPlayer::Black) {
                        HexState::<5>::dsu_union(&mut batch_dsu_black, cell, n_idx);
                    }
                }
            }
            curr = curr.other();
        }

        let black_won = HexState::<5>::dsu_find(&mut batch_dsu_black, 25)
            == HexState::<5>::dsu_find(&mut batch_dsu_black, 26);
        let batch_winner = if black_won {
            HexPlayer::Black
        } else {
            HexPlayer::White
        };

        assert_eq!(
            seq_winner,
            Some(batch_winner),
            "Mismatch between sequential rollout and Black-only batch rollout!"
        );
    }
}

#[test]
fn test_benchmark_batch_vs_sequential_speed() {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use std::time::Instant;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = HexState::<11>::new();
    let iters = 2000;

    // 1. Sequential rollout benchmark
    let start_seq = Instant::now();
    let mut scratch = Vec::with_capacity(121);
    for _ in 0..iters {
        let mut sim = state.clone();
        while !sim.is_won(HexPlayer::Black) && !sim.is_won(HexPlayer::White) {
            sim.legal_actions(&mut scratch);
            if scratch.is_empty() {
                break;
            }
            let choice = *scratch.choose(&mut rng).unwrap();
            sim.play_move(choice);
            sim.current_player = sim.current_player.other();
        }
    }
    let dur_seq = start_seq.elapsed();

    // 2. Batch-fill rollout benchmark
    let start_batch = Instant::now();
    let mut empty_pool: Vec<usize> = (0..121).collect();
    for _ in 0..iters {
        let mut batch_board = state.board.clone();
        let mut batch_dsu_black = state.dsu_black.clone();
        empty_pool.shuffle(&mut rng);

        let mut curr = state.current_player;
        for &cell in &empty_pool {
            batch_board[cell] = Some(curr);
            if curr == HexPlayer::Black {
                let (r, c) = HexState::<11>::coord(cell);
                if r == 0 {
                    HexState::<11>::dsu_union(&mut batch_dsu_black, cell, 121);
                }
                if r == 10 {
                    HexState::<11>::dsu_union(&mut batch_dsu_black, cell, 122);
                }
                for (nr, nc) in HexState::<11>::neighbors(r, c) {
                    let n_idx = HexState::<11>::idx(nr, nc);
                    if batch_board[n_idx] == Some(HexPlayer::Black) {
                        HexState::<11>::dsu_union(&mut batch_dsu_black, cell, n_idx);
                    }
                }
            }
            curr = curr.other();
        }

        let _black_won = HexState::<11>::dsu_find(&mut batch_dsu_black, 121)
            == HexState::<11>::dsu_find(&mut batch_dsu_black, 122);
    }
    let dur_batch = start_batch.elapsed();

    println!(
        "\n[ROLLOUT BENCHMARK (11x11, 2000 rollouts)]:\n  Sequential Rollout: {:?}\n  Batch-Fill Rollout: {:?}\n  Speedup: {:.2}x\n",
        dur_seq,
        dur_batch,
        dur_seq.as_secs_f64() / dur_batch.as_secs_f64()
    );
}

#[test]
fn test_pie_rule_legality_and_transposition() {
    let mut state = HexState::<5>::with_pie_rule(true);
    let mut legals = Vec::new();

    // Turn 1 (Black): SWAP_ACTION is not legal
    state.legal_actions(&mut legals);
    assert!(!legals.contains(&HexState::<5>::SWAP_ACTION));

    // Black plays B1 (row 0, col 1 -> index 1)
    state.play_move(1);
    state.current_player = HexPlayer::White;
    assert_eq!(state.move_count, 1);

    // Turn 2 (White): SWAP_ACTION is legal!
    state.legal_actions(&mut legals);
    assert!(legals.contains(&HexState::<5>::SWAP_ACTION));

    // Execute SWAP_ACTION
    state.play_move(HexState::<5>::SWAP_ACTION);
    assert_eq!(state.move_count, 2);
    assert_eq!(state.current_player, HexPlayer::Black);

    // Stone at (0, 1) was cleared, stone at (1, 0) is now White (A2 -> index 5)
    assert_eq!(state.board[1], None);
    assert_eq!(state.board[5], Some(HexPlayer::White));

    // Turn 3 (Black): SWAP_ACTION must never be legal again
    state.legal_actions(&mut legals);
    assert!(!legals.contains(&HexState::<5>::SWAP_ACTION));
}

#[test]
fn test_pie_rule_disabled_by_default() {
    let mut state = HexState::<5>::new();
    state.play_move(1);
    state.current_player = HexPlayer::White;

    let mut legals = Vec::new();
    state.legal_actions(&mut legals);
    assert!(!legals.contains(&HexState::<5>::SWAP_ACTION));
}

#[test]
fn test_heuristic_agent_invokes_pie_rule_on_center_opening() {
    let mut state = HexState::<5>::with_pie_rule(true);
    // Black opens at C3 (center of 5x5: row 2, col 2 -> index 12)
    state.play_move(12);
    state.current_player = HexPlayer::White;

    let mut agent = HeuristicAgent::<5>::new();
    let action = agent.select_action(&state);
    assert_eq!(
        action,
        HexState::<5>::SWAP_ACTION,
        "HeuristicAgent should invoke the Pie Rule to steal a strong center opening"
    );
}

#[test]
fn test_str_to_coord_swap() {
    assert_eq!(HexState::<11>::str_to_coord("swap"), Some(121));
    assert_eq!(HexState::<11>::str_to_coord("SWAP"), Some(121));
    assert_eq!(HexState::<11>::str_to_coord("  Swap  "), Some(121));
    assert_eq!(HexState::<11>::coord_to_str(121), "swap");
}
