//! Interactive terminal player for Hex.
//!
//! # Usage Examples
//! ```bash
//! # Play as Black against heuristic-guided MCTS (11x11 board)
//! cargo run -p hex --bin hex-play
//!
//! # Play as White against MCTS on a fast 7x7 board
//! cargo run -p hex --bin hex-play -- --size 7 --p1 mcts-h:1000 --p2 human
//!
//! # AI vs AI demonstration (Heuristic vs MCTS)
//! cargo run -p hex --bin hex-play -- --size 7 --p1 heuristic --p2 mcts-h:1000 --verbose
//! ```

use hex::agent::HexAgentSpec;
use hex::game::{HexPlayer, HexState};
use hex::render::render_board_styled;
use hex::world::HexWorld;
use std::env;
use std::time::Instant;

fn print_help() {
    println!(
        r#"Hex - Interactive Terminal Player

USAGE:
    hex-play [OPTIONS]

OPTIONS:
    --board-size <N> (or --size)
                     Board dimension N (supports 7, 9, or 11) [default: 11]
    --p1 <SPEC>      Player 1 (Black, moves first) [default: human]
    --p2 <SPEC>      Player 2 (White, moves second) [default: mcts-h:1000]
    --pie-rule       Enable the Pie (Swap) rule on Move 2 for White
    --verbose        Print MCTS search candidate tables
    -h, --help       Print help information

SUPPORTED AGENT SPECS:
    human                      Interactive keyboard player
    heuristic                  1-ply greedy lookahead (shortest path)
    random                     Uniform random player
    mcts-h[:iters]             MCTS with shortest-path heuristic (recommended)
    mcts[:iters[:rollouts]]    MCTS with Monte Carlo rollouts
    mcts-u[:iters]             MCTS with uniform priors
"#
    );
}

fn play_game<const N: usize>(
    p1_spec: &HexAgentSpec,
    p2_spec: &HexAgentSpec,
    pie_rule: bool,
    verbose: bool,
) {
    let mut p1 = p1_spec.instantiate::<N>(format!("{} (Black)", p1_spec.default_name()), verbose);
    let mut p2 = p2_spec.instantiate::<N>(format!("{} (White)", p2_spec.default_name()), verbose);

    let world = HexWorld::<N>::with_pie_rule(pie_rule);
    let mut state = HexState::<N>::with_pie_rule(pie_rule);

    println!("================================================================================");
    println!("                               HEX ARENA ({}x{})", N, N);
    println!("================================================================================");
    println!("  Black [X] (Top ↔ Bottom):  {}", p1.name());
    println!("  White [O] (Left ↔ Right): {}", p2.name());
    println!(
        "  Pie Rule (Swap):           {}",
        if pie_rule { "Enabled" } else { "Disabled" }
    );
    println!("================================================================================\n");

    let start_time = Instant::now();
    let mut move_count = 0;

    print!("{}", render_board_styled(&state, true));

    while !world.is_terminal(&state) {
        let player = state.current_player;
        let action = match player {
            HexPlayer::Black => p1.select_action(&state),
            HexPlayer::White => p2.select_action(&state),
        };

        if action == HexState::<N>::SWAP_ACTION {
            println!(
                "\n>>> Turn {:>2}: {} invoked the PIE RULE (swapped opening stone to White!)\n",
                move_count + 1,
                player.name()
            );
        } else {
            let coord = HexState::<N>::coord_to_str(action);
            println!(
                "\n>>> Turn {:>2}: {} placed stone at {} (cell {})\n",
                move_count + 1,
                player.name(),
                coord,
                action
            );
        }

        let outcome = world.step_action(&mut state, action);
        move_count += 1;

        print!("{}", render_board_styled(&state, true));

        if outcome.terminated {
            let duration = start_time.elapsed();
            println!(
                "================================================================================"
            );
            if outcome.reward[0] > 0.0 {
                println!(
                    "🏆 GAME OVER: {} (Black [X]) WINS by connecting Top to Bottom!",
                    p1.name()
                );
            } else if outcome.reward[1] > 0.0 {
                println!(
                    "🏆 GAME OVER: {} (White [O]) WINS by connecting Left to Right!",
                    p2.name()
                );
            } else {
                println!("GAME OVER: Board filled with no winner.");
            }
            println!(
                "Total Moves: {} | Elapsed Time: {:.2}s ({:.1} moves/s)",
                move_count,
                duration.as_secs_f32(),
                move_count as f32 / duration.as_secs_f32().max(0.001)
            );
            println!(
                "================================================================================"
            );
            break;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut size: usize = 11;
    let mut p1_str = "human".to_string();
    let mut p2_str = "mcts-h:1000".to_string();
    let mut pie_rule = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "--board-size" | "--size" => {
                i += 1;
                if i < args.len() {
                    size = args[i].parse().unwrap_or(11);
                }
            }
            "--p1" => {
                i += 1;
                if i < args.len() {
                    p1_str = args[i].clone();
                }
            }
            "--p2" => {
                i += 1;
                if i < args.len() {
                    p2_str = args[i].clone();
                }
            }
            "--pie-rule" => {
                pie_rule = true;
            }
            "--verbose" => {
                verbose = true;
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_help();
                return;
            }
        }
        i += 1;
    }

    let p1_spec = match HexAgentSpec::parse(&p1_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error parsing --p1: {e}");
            return;
        }
    };
    let p2_spec = match HexAgentSpec::parse(&p2_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error parsing --p2: {e}");
            return;
        }
    };

    match size {
        7 => play_game::<7>(&p1_spec, &p2_spec, pie_rule, verbose),
        9 => play_game::<9>(&p1_spec, &p2_spec, pie_rule, verbose),
        11 => play_game::<11>(&p1_spec, &p2_spec, pie_rule, verbose),
        other => {
            eprintln!("Unsupported size: {other}. Supported: 7, 9, 11");
        }
    }
}
