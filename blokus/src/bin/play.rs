//! Interactive terminal Blokus game runner.
//!
//! Play against MCTS AI or watch multi-agent battles in Blokus Duo (2-player) or Classic (4-player).

use blokus::agent::{Agent, HeuristicAgent, HumanAgent, MctsAgent};
use blokus::game::{BlokusClassicState, BlokusDuoState, Player};
use blokus::render::{render_board, render_inventory, render_scoreboard};
use std::env;

fn print_help() {
    println!(
        r#"Blokus - Multi-Agent MCTS Terminal Runner

USAGE:
    blokus-play [OPTIONS]

OPTIONS:
    --variant <VAR>    Game variant: 'duo' (14x14, 2P) [default] or 'classic' (20x20, 4P)
    --mode <MODE>      For Duo: 'human-ai' [default], 'ai-human', 'ai-ai', 'human-human', 'ai-heuristic'
                       For Classic: 'human-3ai' [default], '4ai'
    --iters <N>        MCTS search iterations per move [default: 400]
    --no-color         Disable ANSI terminal colors
    --quiet            Do not print detailed MCTS move statistics
    -h, --help         Print help information
"#
    );
}

fn run_duo(mode: &str, iters: usize, verbose: bool, use_color: bool) {
    let mut state = BlokusDuoState::new();

    let mut p0: Box<dyn Agent<14, 2>> = match mode {
        "ai-human" => Box::new(MctsAgent::new_heuristic("MCTS AI (Blue)", iters, verbose)),
        "ai-ai" => Box::new(MctsAgent::new_heuristic(
            "MCTS Alpha (Blue)",
            iters,
            verbose,
        )),
        "human-ai" | "human-human" => Box::new(HumanAgent::new("Human (Blue)")),
        _ => Box::new(HumanAgent::new("Human (Blue)")),
    };

    let mut p1: Box<dyn Agent<14, 2>> = match mode {
        "human-ai" => Box::new(MctsAgent::new_heuristic("MCTS AI (Yellow)", iters, verbose)),
        "ai-human" | "human-human" => Box::new(HumanAgent::new("Human (Yellow)")),
        "ai-ai" => Box::new(MctsAgent::new_heuristic(
            "MCTS Beta (Yellow)",
            iters,
            verbose,
        )),
        "ai-heuristic" => Box::new(HeuristicAgent::new("Heuristic Bot (Yellow)")),
        _ => Box::new(MctsAgent::new_heuristic("MCTS AI (Yellow)", iters, verbose)),
    };

    println!("============================================================");
    println!("             🔷 BLOKUS DUO - MULTI-AGENT MCTS 🔶           ");
    println!("============================================================");
    println!("Mode: {mode} | Variant: 14x14 Duo | Iterations: {iters}\n");

    while !state.is_terminal() {
        print!("{}", render_board(&state, use_color));
        print!("{}", render_scoreboard(&state, use_color));

        let current_p = state.current_player as usize;
        let is_human = (current_p == 0 && (mode == "human-ai" || mode == "human-human"))
            || (current_p == 1 && (mode == "ai-human" || mode == "human-human"));

        if is_human {
            print!("{}", render_inventory(&state, current_p));
        }

        let action = if current_p == 0 {
            p0.select_action(&state)
        } else {
            p1.select_action(&state)
        };

        println!(
            "\n>>> {} ({}) plays: {}",
            if current_p == 0 { p0.name() } else { p1.name() },
            Player::from_index(current_p).name(),
            action
        );

        if let Err(err) = state.apply_action(&action) {
            println!("Move error: {err}. Ending game.");
            break;
        }
    }

    println!("\n=================== GAME OVER ===================");
    print!("{}", render_board(&state, use_color));
    print!("{}", render_scoreboard(&state, use_color));

    let s0 = state.score(0);
    let s1 = state.score(1);
    if s0 > s1 {
        println!("🏆 Winner: Player 0 (Blue) with score {s0} vs {s1}!");
    } else if s1 > s0 {
        println!("🏆 Winner: Player 1 (Yellow) with score {s1} vs {s0}!");
    } else {
        println!("🤝 Draw! Both players tied at score {s0}.");
    }
}

fn run_classic(mode: &str, iters: usize, verbose: bool, use_color: bool) {
    let mut state = BlokusClassicState::new();

    let mut players: [Box<dyn Agent<20, 4>>; 4] = match mode {
        "human-3ai" => [
            Box::new(HumanAgent::new("Human (Blue)")),
            Box::new(MctsAgent::new_heuristic("MCTS-1 (Yellow)", iters, verbose)),
            Box::new(MctsAgent::new_heuristic("MCTS-2 (Red)", iters, verbose)),
            Box::new(MctsAgent::new_heuristic("MCTS-3 (Green)", iters, verbose)),
        ],
        _ => [
            Box::new(MctsAgent::new_heuristic("MCTS-0 (Blue)", iters, verbose)),
            Box::new(MctsAgent::new_heuristic("MCTS-1 (Yellow)", iters, verbose)),
            Box::new(MctsAgent::new_heuristic("MCTS-2 (Red)", iters, verbose)),
            Box::new(MctsAgent::new_heuristic("MCTS-3 (Green)", iters, verbose)),
        ],
    };

    println!("============================================================");
    println!("        🔷 BLOKUS CLASSIC - 4-PLAYER VECTOR MCTS 🔶         ");
    println!("============================================================");
    println!("Mode: {mode} | Variant: 20x20 Classic | Iterations: {iters}\n");

    while !state.is_terminal() {
        print!("{}", render_board(&state, use_color));
        print!("{}", render_scoreboard(&state, use_color));

        let current_p = state.current_player as usize;
        let is_human = current_p == 0 && mode == "human-3ai";
        if is_human {
            print!("{}", render_inventory(&state, current_p));
        }

        let action = players[current_p].select_action(&state);
        println!(
            "\n>>> {} ({}) plays: {}",
            players[current_p].name(),
            Player::from_index(current_p).name(),
            action
        );

        if let Err(err) = state.apply_action(&action) {
            println!("Move error: {err}. Ending game.");
            break;
        }
    }

    println!("\n=================== GAME OVER ===================");
    print!("{}", render_board(&state, use_color));
    print!("{}", render_scoreboard(&state, use_color));

    let scores = [
        state.score(0),
        state.score(1),
        state.score(2),
        state.score(3),
    ];
    let max_score = *scores.iter().max().unwrap();
    let winners: Vec<usize> = (0..4).filter(|&p| scores[p] == max_score).collect();
    if winners.len() == 1 {
        let w = winners[0];
        println!(
            "🏆 Winner: Player {} ({}) with score {max_score}!",
            w,
            Player::from_index(w).name()
        );
    } else {
        println!(
            "🤝 Multi-way tie for 1st place with score {max_score}: {:?}",
            winners
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut variant = "duo".to_string();
    let mut mode = "human-ai".to_string();
    let mut iters = 400;
    let mut use_color = true;
    let mut verbose = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--variant" => {
                if i + 1 < args.len() {
                    variant = args[i + 1].clone();
                    i += 1;
                }
            }
            "--mode" => {
                if i + 1 < args.len() {
                    mode = args[i + 1].clone();
                    i += 1;
                }
            }
            "--iters" => {
                if i + 1 < args.len() {
                    iters = args[i + 1].parse().unwrap_or(400);
                    i += 1;
                }
            }
            "--no-color" => use_color = false,
            "--quiet" => verbose = false,
            "-h" | "--help" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    if variant == "classic" {
        if mode == "human-ai" {
            mode = "human-3ai".to_string();
        }
        run_classic(&mode, iters, verbose, use_color);
    } else {
        run_duo(&mode, iters, verbose, use_color);
    }
}
