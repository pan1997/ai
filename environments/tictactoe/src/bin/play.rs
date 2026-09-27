//! Interactive terminal TicTacToe game runner.

use std::env;
use tictactoe::agent::{BoxAgent, parse_agent_spec};
use tictactoe::game::{Player, TicTacToeState};
use tictactoe::render::render_board_styled;

fn print_help() {
    println!(
        r#"TicTacToe - Interactive Terminal Runner

USAGE:
    tictactoe-play [OPTIONS]

OPTIONS:
    --p1 <SPEC>       Player 1 (X) spec: 'human', 'random', 'tactical', 'mcts:<sims>' [default: human]
    --p2 <SPEC>       Player 2 (O) spec: 'human', 'random', 'tactical', 'mcts:<sims>' [default: mcts:100]
    --no-color        Disable ANSI terminal colors
    -h, --help        Print help information
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut p1_spec = "human".to_string();
    let mut p2_spec = "mcts:100".to_string();
    let mut use_color = true;

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
            "--no-color" => {
                use_color = false;
            }
            _ => {
                eprintln!("Unknown argument '{}'. Use --help for usage.", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut agent_x: BoxAgent = parse_agent_spec(&p1_spec)
        .unwrap_or_else(|e| panic!("Failed to parse p1 spec: {e}"));
    let mut agent_o: BoxAgent = parse_agent_spec(&p2_spec)
        .unwrap_or_else(|e| panic!("Failed to parse p2 spec: {e}"));

    println!("Starting TicTacToe: {} (X) vs {} (O)", agent_x.name(), agent_o.name());
    let mut state = TicTacToeState::new();

    while !state.is_terminal() {
        println!("\n{}", render_board_styled(&state, use_color));
        let action = match state.current_player {
            Player::X => agent_x.select_action(&state),
            Player::O => agent_o.select_action(&state),
        };
        println!("{} plays cell {action}", state.current_player.symbol());
        state.apply_action(action);
    }

    println!("\nGame Over!");
    println!("{}", render_board_styled(&state, use_color));

    match state.check_winner() {
        Some(Player::X) => println!("Player X ({}) wins!", agent_x.name()),
        Some(Player::O) => println!("Player O ({}) wins!", agent_o.name()),
        None => println!("It's a draw!"),
    }
}
