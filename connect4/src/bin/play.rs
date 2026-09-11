//! Interactive terminal Connect 4 game runner.
//!
//! Play against an MCTS AI, pit two AIs against each other, or play locally against a friend.
//!
//! # Usage
//! ```bash
//! cargo run -p connect4 --bin connect4-play -- [OPTIONS]
//!
//! Options:
//!   --mode <MODE>       Game mode: 'human-ai', 'ai-human', 'ai-ai', 'human-human' [default: human-ai]
//!   --iters <N>         MCTS search iterations per move [default: 800]
//!   --rollouts <N>      Rollout evaluations per leaf (0 for uniform prior) [default: 5]
//!   --depth <N>         Max rollout simulation depth [default: 20]
//!   --no-color          Disable ANSI colored board rendering
//!   --quiet             Hide detailed MCTS candidate move stats
//!   -h, --help          Print this help message
//! ```

use connect4::agent::{Agent, HumanAgent, MctsAgent, RandomAgent};
use connect4::game::{Connect4State, Player};
use connect4::render::render_board_styled;
use std::env;

fn print_help() {
    println!(
        r#"Connect 4 - Interactive Terminal Runner

USAGE:
    connect4-play [OPTIONS]

OPTIONS:
    --mode <MODE>     Game mode:
                        'human-ai'    Human (Red) vs MCTS (Yellow) [default]
                        'ai-human'    MCTS (Red) vs Human (Yellow)
                        'ai-ai'       MCTS (Red) vs MCTS (Yellow)
                        'human-human' Local 2-Player Pass & Play
                        'ai-random'   MCTS (Red) vs Random Bot (Yellow)
    --iters <N>       MCTS search iterations per move [default: 800]
    --rollouts <N>    Rollouts per leaf evaluation (0 for uniform prior) [default: 5]
    --depth <N>       Max simulation depth during rollouts [default: 20]
    --no-color        Disable ANSI terminal colors
    --quiet           Do not print detailed MCTS search statistics
    -h, --help        Print help information
"#
    );
}

fn build_mcts_agent(
    name: &str,
    iters: usize,
    rollouts: usize,
    depth: usize,
    verbose: bool,
) -> Box<dyn Agent<6, 7>> {
    if rollouts == 0 {
        Box::new(MctsAgent::new_uniform(name, iters, verbose))
    } else {
        Box::new(MctsAgent::new_rollout(name, iters, rollouts, depth, verbose))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut mode = "human-ai".to_string();
    let mut iters = 800;
    let mut rollouts = 5;
    let mut depth = 20;
    let mut use_color = true;
    let mut verbose = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "--mode" => {
                i += 1;
                if i < args.len() {
                    mode = args[i].clone();
                }
            }
            "--iters" => {
                i += 1;
                if i < args.len() {
                    iters = args[i].parse().unwrap_or(800);
                }
            }
            "--rollouts" => {
                i += 1;
                if i < args.len() {
                    rollouts = args[i].parse().unwrap_or(5);
                }
            }
            "--depth" => {
                i += 1;
                if i < args.len() {
                    depth = args[i].parse().unwrap_or(20);
                }
            }
            "--no-color" => {
                use_color = false;
            }
            "--quiet" => {
                verbose = false;
            }
            other => {
                eprintln!("Unknown option '{other}'. Run with --help for usage.");
                return;
            }
        }
        i += 1;
    }

    println!("==================================================");
    println!("        🔴 CONNECT 4 - ZERO-ALLOCATION MCTS 🟡    ");
    println!("==================================================");
    println!("Mode: {mode} | MCTS Iterations: {iters} | Rollouts: {rollouts}\n");

    let (mut player_red, mut player_yellow): (Box<dyn Agent<6, 7>>, Box<dyn Agent<6, 7>>) =
        match mode.as_str() {
            "human-ai" => (
                Box::new(HumanAgent::new("You (Red)")),
                build_mcts_agent("MCTS Bot (Yellow)", iters, rollouts, depth, verbose),
            ),
            "ai-human" => (
                build_mcts_agent("MCTS Bot (Red)", iters, rollouts, depth, verbose),
                Box::new(HumanAgent::new("You (Yellow)")),
            ),
            "ai-ai" => (
                build_mcts_agent("Alpha-MCTS (Red)", iters, rollouts, depth, verbose),
                build_mcts_agent("Beta-MCTS (Yellow)", iters, rollouts, depth, verbose),
            ),
            "human-human" => (
                Box::new(HumanAgent::new("Player 1 (Red)")),
                Box::new(HumanAgent::new("Player 2 (Yellow)")),
            ),
            "ai-random" => (
                build_mcts_agent("MCTS Bot (Red)", iters, rollouts, depth, verbose),
                Box::new(RandomAgent::new("Random Bot (Yellow)")),
            ),
            other => {
                eprintln!("Invalid mode '{other}'. Supported: human-ai, ai-human, ai-ai, human-human, ai-random");
                return;
            }
        };

    let mut state = Connect4State::<6, 7>::new();

    loop {
        println!("{}", render_board_styled(&state, use_color));

        let (active_name, active_agent) = match state.current_player {
            Player::Red => (player_red.name().to_string(), &mut player_red),
            Player::Yellow => (player_yellow.name().to_string(), &mut player_yellow),
        };

        println!("Turn: {active_name} [{:?}]", state.current_player);

        let chosen_col = active_agent.select_action(&state);
        let current_player = state.current_player;

        let placed_row = state.drop_piece(chosen_col).unwrap_or_else(|err| {
            panic!("Agent {active_name} selected invalid column {chosen_col}: {err}")
        });

        println!("{active_name} dropped checker into column {chosen_col}.\n");

        if state.check_win_at(placed_row, chosen_col, current_player) {
            println!("{}", render_board_styled(&state, use_color));
            println!("🎉🎉 Game Over! {active_name} [{:?}] wins! 🎉🎉\n", current_player);
            break;
        }

        if state.is_board_full() {
            println!("{}", render_board_styled(&state, use_color));
            println!("🤝 Game Over! The board is full — it's a draw! 🤝\n");
            break;
        }

        state.current_player = current_player.other();
    }
}

