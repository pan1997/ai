//! Interactive 2048 game runner supporting human play and AI watch mode.

use mcts_traits::{Agent, World};
use std::env;
use std::thread::sleep;
use std::time::Duration;
use tzf8::agent::Tzf8AgentSpec;
use tzf8::render::render_board;
use tzf8::world::Tzf8World;

fn print_usage() {
    println!("2048 Interactive Runner - Play manually or watch AI agents play in real time");
    println!();
    println!("Usage: cargo run --release -p tzf8 --bin tzf8-play -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --agent <spec>     Agent to run: 'human', 'heuristic', 'mcts[:iters]', 'random'");
    println!("                     (default: 'human')");
    println!(
        "  --delay <ms>       Delay between AI moves in milliseconds (default: 150 for AI, 0 for human)"
    );
    println!("  --seed <S>         Random seed for reproducible initial board");
    println!("  --help             Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut agent_spec_str = "human".to_string();
    let mut delay_ms: Option<u64> = None;
    let mut seed: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agent" => {
                if i + 1 < args.len() {
                    agent_spec_str = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --agent requires a value");
                    std::process::exit(1);
                }
            }
            "--delay" => {
                if i + 1 < args.len() {
                    delay_ms = Some(args[i + 1].parse().expect("Invalid --delay number"));
                    i += 2;
                } else {
                    eprintln!("Error: --delay requires a value");
                    std::process::exit(1);
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = Some(args[i + 1].parse().expect("Invalid --seed number"));
                    i += 2;
                } else {
                    eprintln!("Error: --seed requires a value");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("Unknown option '{other}'");
                print_usage();
                std::process::exit(1);
            }
        }
    }

    let spec = Tzf8AgentSpec::parse(&agent_spec_str)
        .unwrap_or_else(|e| panic!("Error parsing agent '{agent_spec_str}': {e}"));

    let is_human = matches!(spec, Tzf8AgentSpec::Human);
    let effective_delay = delay_ms.unwrap_or(if is_human { 0 } else { 150 });

    let mut agent = spec.build_agent();
    let world = match seed {
        Some(s) => Tzf8World::with_seed(s),
        None => Tzf8World::new(),
    };
    let mut state = match seed {
        Some(s) => Tzf8World::initial_with_seed(s),
        None => world.initial(),
    };

    println!("=================================================================");
    println!("                      2048 INTERACTIVE PLAY                      ");
    println!("=================================================================");
    println!("Player: {}", agent.name());
    println!();

    let mut move_count = 0;

    while !world.is_terminal(&state) {
        println!("{}", render_board(&state));

        if !is_human && effective_delay > 0 {
            sleep(Duration::from_millis(effective_delay));
        }

        let action = agent.select_action(&state);
        move_count += 1;
        println!(">>> Move #{}: {}\n", move_count, action.name());

        let outcome = world.step_action(&mut state, action);
        if outcome.terminated {
            break;
        }
    }

    println!("=================================================================");
    println!("                           GAME OVER                             ");
    println!("=================================================================");
    println!("{}", render_board(&state));
    println!(
        "Final Score: {} | Max Tile: {} | Total Moves: {}",
        state.score,
        state.max_tile(),
        move_count
    );
}
