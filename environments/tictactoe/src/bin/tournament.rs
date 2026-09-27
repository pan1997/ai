//! Tournament arena runner for TicTacToe.

use mcts_engine::arena::{MatchDriver, TwoPlayerTournamentStats};
use std::env;
use std::time::Instant;
use tictactoe::TicTacToeWorld;
use tictactoe::agent::parse_agent_spec;

fn print_help() {
    println!(
        r#"TicTacToe - Tournament Runner

USAGE:
    tictactoe-tournament [OPTIONS]

OPTIONS:
    --p1 <SPEC>       Player 1 agent spec [default: mcts:100]
    --p2 <SPEC>       Player 2 agent spec [default: random]
    --games <N>       Number of games per matchup [default: 50] (half as X, half as O)
    -h, --help        Print help information
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut p1_spec = "mcts:100".to_string();
    let mut p2_spec = "random".to_string();
    let mut num_games = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "--p1" => {
                i += 1;
                p1_spec = args[i].clone();
            }
            "--p2" => {
                i += 1;
                p2_spec = args[i].clone();
            }
            "--games" => {
                i += 1;
                num_games = args[i].parse::<usize>().expect("Invalid number of games");
            }
            _ => {
                eprintln!("Unknown argument '{}'. Use --help for usage.", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let world = TicTacToeWorld::new();
    let mut stats = vec![TwoPlayerTournamentStats::default(), TwoPlayerTournamentStats::default()];
    let names = vec![p1_spec.clone(), p2_spec.clone()];
    let start_time = Instant::now();

    println!("Running tournament: {} vs {} ({} games)", p1_spec, p2_spec, num_games);

    for game_idx in 0..num_games {
        // Seat balancing: alternate who plays X (seat 0) and O (seat 1)
        let (mut agent0, mut agent1, p1_is_seat0) = if game_idx % 2 == 0 {
            (
                parse_agent_spec(&p1_spec).unwrap(),
                parse_agent_spec(&p2_spec).unwrap(),
                true,
            )
        } else {
            (
                parse_agent_spec(&p2_spec).unwrap(),
                parse_agent_spec(&p1_spec).unwrap(),
                false,
            )
        };

        let driver = MatchDriver::new();
        let result = driver.play_2p(&world, &mut *agent0, &mut *agent1, None);

        let (seat0_agent, seat1_agent) = if p1_is_seat0 { (0, 1) } else { (1, 0) };
        let moves0 = result.moves_per_seat[0];
        let moves1 = result.moves_per_seat[1];

        TwoPlayerTournamentStats::record_game(
            &mut stats,
            seat0_agent,
            seat1_agent,
            result.outcome_2p.unwrap(),
            moves0,
            moves1,
        );
    }

    let duration = start_time.elapsed();
    println!("\nTournament Completed in {:.2?}!", duration);
    TwoPlayerTournamentStats::print_standings(&stats, &names, "X", "O");
}
