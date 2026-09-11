//! Benchmark arena pitting two agents against each other across $N$ games.
//!
//! Alternates first-player colors across games to balance first-mover advantage.
//!
//! # Usage
//! ```bash
//! cargo run -p connect4 --bin connect4-tournament -- [OPTIONS]
//!
//! Options:
//!   --games <N>           Total number of games [default: 10]
//!   --mcts-iters <N>      MCTS iterations for Agent 1 [default: 200]
//!   --opponent <TYPE>     Opponent type: 'random' or 'mcts' [default: random]
//!   --opponent-iters <N>  MCTS iterations for Agent 2 (if opponent is mcts) [default: 50]
//!   --rollouts <N>        Rollouts per leaf evaluation [default: 3]
//!   -h, --help            Print help information
//! ```

use connect4::agent::{Agent, MctsAgent, RandomAgent};
use connect4::game::{Connect4State, Player};
use std::env;
use std::time::Instant;

fn print_help() {
    println!(
        r#"Connect 4 - Tournament Benchmark Arena

USAGE:
    connect4-tournament [OPTIONS]

OPTIONS:
    --games <N>           Total number of tournament games [default: 10]
    --mcts-iters <N>      MCTS iterations for Agent 1 [default: 200]
    --opponent <TYPE>     Opponent type: 'random' or 'mcts' [default: random]
    --opponent-iters <N>  MCTS iterations for Agent 2 (if opponent is mcts) [default: 50]
    --rollouts <N>        Rollout evaluations per leaf (0 for uniform) [default: 3]
    -h, --help            Print help information
"#
    );
}

fn build_agent(
    name: &str,
    agent_type: &str,
    iters: usize,
    rollouts: usize,
) -> Box<dyn Agent<6, 7>> {
    match agent_type {
        "random" => Box::new(RandomAgent::new(name)),
        "mcts" => {
            if rollouts == 0 {
                Box::new(MctsAgent::new_uniform(name, iters, false))
            } else {
                Box::new(MctsAgent::new_rollout(name, iters, rollouts, 20, false))
            }
        }
        other => panic!("Unknown agent type '{other}'"),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut total_games = 10;
    let mut a1_iters = 200;
    let mut opponent_type = "random".to_string();
    let mut opponent_iters = 50;
    let mut rollouts = 3;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "--games" => {
                i += 1;
                if i < args.len() {
                    total_games = args[i].parse().unwrap_or(10);
                }
            }
            "--mcts-iters" => {
                i += 1;
                if i < args.len() {
                    a1_iters = args[i].parse().unwrap_or(200);
                }
            }
            "--opponent" => {
                i += 1;
                if i < args.len() {
                    opponent_type = args[i].clone();
                }
            }
            "--opponent-iters" => {
                i += 1;
                if i < args.len() {
                    opponent_iters = args[i].parse().unwrap_or(50);
                }
            }
            "--rollouts" => {
                i += 1;
                if i < args.len() {
                    rollouts = args[i].parse().unwrap_or(3);
                }
            }
            other => {
                eprintln!("Unknown option '{other}'. Run with --help for usage.");
                return;
            }
        }
        i += 1;
    }

    let a1_name = format!("Agent1-MCTS({a1_iters})");
    let a2_name = if opponent_type == "mcts" {
        format!("Agent2-MCTS({opponent_iters})")
    } else {
        "Agent2-Random".to_string()
    };

    println!("==================================================");
    println!("        ⚔️  CONNECT 4 TOURNAMENT ARENA ⚔️          ");
    println!("==================================================");
    println!("Games: {total_games}");
    println!("Player 1: {a1_name}");
    println!("Player 2: {a2_name}");
    println!("Rollouts: {rollouts}\n");

    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut draws = 0;
    let mut total_moves = 0;

    let start_time = Instant::now();

    for game_idx in 0..total_games {
        // Alternate who plays Red (first move)
        let p1_is_red = game_idx % 2 == 0;

        let (mut red_agent, mut yellow_agent) = if p1_is_red {
            (
                build_agent(&a1_name, "mcts", a1_iters, rollouts),
                build_agent(&a2_name, &opponent_type, opponent_iters, rollouts),
            )
        } else {
            (
                build_agent(&a2_name, &opponent_type, opponent_iters, rollouts),
                build_agent(&a1_name, "mcts", a1_iters, rollouts),
            )
        };

        let mut state = Connect4State::<6, 7>::new();
        let mut moves = 0;

        let game_result = loop {
            let active_agent = match state.current_player {
                Player::Red => &mut red_agent,
                Player::Yellow => &mut yellow_agent,
            };

            let col = active_agent.select_action(&state);
            let current_player = state.current_player;
            let placed_row = state
                .drop_piece(col)
                .unwrap_or_else(|e| panic!("Tournament: invalid move {col}: {e}"));
            moves += 1;

            if state.check_win_at(placed_row, col, current_player) {
                break Some(current_player);
            }
            if state.is_board_full() {
                break None;
            }
            state.current_player = current_player.other();
        };

        total_moves += moves;

        match game_result {
            Some(Player::Red) => {
                if p1_is_red {
                    p1_wins += 1;
                    println!("Game #{:>2}: {} (Red) won in {moves} moves", game_idx + 1, a1_name);
                } else {
                    p2_wins += 1;
                    println!("Game #{:>2}: {} (Red) won in {moves} moves", game_idx + 1, a2_name);
                }
            }
            Some(Player::Yellow) => {
                if p1_is_red {
                    p2_wins += 1;
                    println!("Game #{:>2}: {} (Yellow) won in {moves} moves", game_idx + 1, a2_name);
                } else {
                    p1_wins += 1;
                    println!("Game #{:>2}: {} (Yellow) won in {moves} moves", game_idx + 1, a1_name);
                }
            }
            None => {
                draws += 1;
                println!("Game #{:>2}: Draw in {moves} moves", game_idx + 1);
            }
        }
    }

    let elapsed = start_time.elapsed();
    let moves_per_sec = total_moves as f64 / elapsed.as_secs_f64().max(0.001);

    println!("\n==================================================");
    println!("               TOURNAMENT RESULTS                 ");
    println!("==================================================");
    println!("Total Games:      {total_games}");
    println!(
        "{} Wins: {:>3} ({:>5.1}%)",
        a1_name,
        p1_wins,
        (p1_wins as f64 / total_games as f64) * 100.0
    );
    println!(
        "{} Wins: {:>3} ({:>5.1}%)",
        a2_name,
        p2_wins,
        (p2_wins as f64 / total_games as f64) * 100.0
    );
    println!(
        "Draws:             {:>3} ({:>5.1}%)",
        draws,
        (draws as f64 / total_games as f64) * 100.0
    );
    println!("Avg Moves / Game: {:>5.1}", total_moves as f64 / total_games as f64);
    println!("Elapsed Time:     {:.2?}", elapsed);
    println!("Throughput:       {:.1} moves/sec", moves_per_sec);
    println!("==================================================");
}

