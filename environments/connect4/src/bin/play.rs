//! Interactive terminal Connect 4 game runner.
//!
//! Play against an AlphaZero neural network agent, an MCTS rollout bot, or locally against a friend.
//!
//! # Usage Examples
//! ```bash
//! # 1. Play as Red (first) against the trained AlphaZero model
//! cargo run --release -p connect4 --bin connect4-play -- --model-path ./models/connect4_latest.onnx --sims 100
//!
//! # 2. Play as Yellow (second, AI moves first) against AlphaZero
//! cargo run --release -p connect4 --bin connect4-play -- --mode ai-human --model-path ./models/connect4_latest.onnx --sims 100
//!
//! # 3. Explicit player specs (arbitrary pairings)
//! cargo run --release -p connect4 --bin connect4-play -- --p1 human --p2 alphazero:./models/connect4_latest.onnx:100
//! ```

use connect4::agent::Connect4AgentSpec;
use connect4::game::{Connect4State, Player};
use connect4::render::render_board_styled;
use std::env;

fn print_help() {
    println!(
        r#"Connect 4 - Interactive Terminal Runner

USAGE:
    connect4-play [OPTIONS]

OPTIONS:
    --board <RxC>         Board dimensions: '6x7', '7x8', '7x9', '8x8', '11x15', '11x19' [default: 6x7]
    --p1 <SPEC>           Player 1 (Red) spec: 'human', 'alphazero:<path>[:sims]', 'mcts[:iters[:rollouts]]', 'tactical', 'random'
    --p2 <SPEC>           Player 2 (Yellow) spec: 'human', 'alphazero:<path>[:sims]', 'mcts[:iters[:rollouts]]', 'tactical', 'random'
    --mode <MODE>         Game mode shorthand when --p1/--p2 omitted:
                            'human-ai'    Human (Red) vs AI (Yellow) [default]
                            'ai-human'    AI (Red) vs Human (Yellow)
                            'ai-ai'       AI (Red) vs AI (Yellow)
                            'human-human' Local 2-Player Pass & Play
                            'ai-random'   AI (Red) vs Random Bot (Yellow)
    --model-path <PATH>   Path to ONNX AlphaZero model. If specified, AI uses AlphaZero.
    --sims <N>            AlphaZero MCTS simulations per move [default: 100]
    --iters <N>           Rollout MCTS search iterations per move [default: 800]
    --rollouts <N>        Rollouts per leaf evaluation (0 for uniform prior) [default: 5]
    --c-puct <FLOAT>      PUCT exploration constant [default: 1.414]
    --no-color            Disable ANSI terminal colors
    --quiet               Do not print detailed MCTS search statistics
    -h, --help            Print help information
"#
    );
}

fn run_play<const R: usize, const C: usize>(
    p1_spec: &Connect4AgentSpec,
    p2_spec: &Connect4AgentSpec,
    c_puct: f32,
    use_color: bool,
    verbose: bool,
) {
    println!("==================================================");
    println!("        🔴 CONNECT 4 - ZERO-ALLOCATION MCTS 🟡    ");
    println!("==================================================");
    println!(
        "Board: {R}x{C} | Red (P1): {} | Yellow (P2): {}\n",
        p1_spec.display_name(),
        p2_spec.display_name()
    );

    let mut player_red = p1_spec.instantiate::<R, C>("Player 1 (Red)", c_puct, verbose);
    let mut player_yellow = p2_spec.instantiate::<R, C>("Player 2 (Yellow)", c_puct, verbose);

    let mut state = Connect4State::<R, C>::new();

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
            println!(
                "🎉🎉 Game Over! {active_name} [{:?}] wins! 🎉🎉\n",
                current_player
            );
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

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut board = "6x7".to_string();
    let mut mode = "human-ai".to_string();
    let mut p1_opt: Option<String> = None;
    let mut p2_opt: Option<String> = None;
    let mut model_path_opt: Option<String> = None;
    let mut sims = 100;
    let mut iters = 800;
    let mut rollouts = 5;
    let mut c_puct = 1.414;
    let mut use_color = true;
    let mut verbose = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "--board" => {
                i += 1;
                if i < args.len() {
                    board = args[i].clone();
                }
            }
            "--p1" => {
                i += 1;
                if i < args.len() {
                    p1_opt = Some(args[i].clone());
                }
            }
            "--p2" => {
                i += 1;
                if i < args.len() {
                    p2_opt = Some(args[i].clone());
                }
            }
            "--model-path" | "--model" => {
                i += 1;
                if i < args.len() {
                    model_path_opt = Some(args[i].clone());
                }
            }
            "--sims" => {
                i += 1;
                if i < args.len() {
                    sims = args[i].parse().unwrap_or(100);
                }
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
            "--c-puct" => {
                i += 1;
                if i < args.len() {
                    c_puct = args[i].parse().unwrap_or(1.414);
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

    let default_ai_spec = match model_path_opt {
        Some(path) => Connect4AgentSpec::AlphaZero {
            model_path: path,
            sims,
        },
        None => Connect4AgentSpec::Mcts { iters, rollouts },
    };

    let p1_spec = match p1_opt {
        Some(s) => match Connect4AgentSpec::parse(&s, iters, rollouts) {
            Ok(spec) => spec,
            Err(e) => {
                eprintln!("Error in --p1 spec: {e}");
                return;
            }
        },
        None => match mode.as_str() {
            "human-ai" | "human-human" => Connect4AgentSpec::Human,
            "ai-human" | "ai-ai" | "ai-random" => default_ai_spec.clone(),
            other => {
                eprintln!(
                    "Invalid mode '{other}'. Supported: human-ai, ai-human, ai-ai, human-human, ai-random"
                );
                return;
            }
        },
    };

    let p2_spec = match p2_opt {
        Some(s) => match Connect4AgentSpec::parse(&s, iters, rollouts) {
            Ok(spec) => spec,
            Err(e) => {
                eprintln!("Error in --p2 spec: {e}");
                return;
            }
        },
        None => match mode.as_str() {
            "human-ai" | "ai-ai" => default_ai_spec,
            "ai-human" | "human-human" => Connect4AgentSpec::Human,
            "ai-random" => Connect4AgentSpec::Random,
            _ => default_ai_spec,
        },
    };

    match board.as_str() {
        "6x7" => run_play::<6, 7>(&p1_spec, &p2_spec, c_puct, use_color, verbose),
        "7x8" => run_play::<7, 8>(&p1_spec, &p2_spec, c_puct, use_color, verbose),
        "7x9" => run_play::<7, 9>(&p1_spec, &p2_spec, c_puct, use_color, verbose),
        "8x8" => run_play::<8, 8>(&p1_spec, &p2_spec, c_puct, use_color, verbose),
        "11x15" => run_play::<11, 15>(&p1_spec, &p2_spec, c_puct, use_color, verbose),
        "11x19" => run_play::<11, 19>(&p1_spec, &p2_spec, c_puct, use_color, verbose),
        other => {
            eprintln!(
                "Invalid board size '{other}'. Supported sizes: 6x7, 7x8, 7x9, 8x8, 11x15, 11x19"
            );
        }
    }
}
