//! Interactive terminal Sequence game runner.
//!
//! Play against AI agents (ISMCTS, Opponent-Model MCTS, Heuristic, Random) or watch AI vs AI
//! matches across 2 to 6 players and 2 or 3 teams.

use sequence::agent::{parse_agent, BoxAgent};
use sequence::game::{SequenceConfig, SequenceState};
use sequence::render::{format_action, render_state, TEAM_NAMES};
use std::env;

fn print_help() {
    println!(
        r#"Sequence - Board & Card Strategy Game Terminal Runner

USAGE:
    sequence-play [OPTIONS]

OPTIONS:
    --players <N>       Number of players (2..=6) [default: 2]
    --teams <T>         Number of teams (2 or 3) [default: 2]
    --mode <MODE>       Quick preset:
                          'human-ai' (2P Human vs ISMCTS) [default]
                          'ai-ai' (2P ISMCTS vs Heuristic)
                          'human-human' (2P pass-and-play)
                          '4p-teams' (4P: 2 teams of 2, Human + AI vs 2 AIs)
                          '3p' (3P: 3 teams, Human vs 2 AIs)
    --p0, --p1, ...     Agent spec for each seat:
                          'human'
                          'random'
                          'heuristic'
                          'mcts:<iters>'
                          'is-mcts:<iters>:<dets>'
                          'macro-heuristic:<iters>:<dets>'
                          'macro-random:<iters>:<dets>'
    --iters <N>         Default MCTS iterations [default: 300]
    --no-color          Disable ANSI colors
    -h, --help          Print help information
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return;
    }

    let mut num_players = 2usize;
    let mut num_teams = 2usize;
    let mut iters = 300usize;
    let mut use_color = true;
    let mut mode = "human-ai".to_string();
    let mut seat_specs: Vec<Option<String>> = vec![None; 6];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--players" if i + 1 < args.len() => {
                num_players = args[i + 1].parse().unwrap_or(2).clamp(2, 6);
                i += 2;
            }
            "--teams" if i + 1 < args.len() => {
                num_teams = args[i + 1].parse().unwrap_or(2).clamp(2, 3);
                i += 2;
            }
            "--iters" if i + 1 < args.len() => {
                iters = args[i + 1].parse().unwrap_or(300);
                i += 2;
            }
            "--mode" if i + 1 < args.len() => {
                mode = args[i + 1].clone();
                i += 2;
            }
            "--no-color" => {
                use_color = false;
                i += 1;
            }
            arg if arg.starts_with("--p") && arg.len() == 4 => {
                if let Ok(seat) = arg[3..4].parse::<usize>()
                    && seat < 6
                    && i + 1 < args.len()
                {
                    seat_specs[seat] = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Apply mode presets if seat specs were not explicitly given
    match mode.as_str() {
        "ai-ai" => {
            num_players = 2;
            num_teams = 2;
            seat_specs[0].get_or_insert_with(|| format!("is-mcts:{iters}:5"));
            seat_specs[1].get_or_insert_with(|| "heuristic".to_string());
        }
        "human-human" => {
            num_players = 2;
            num_teams = 2;
            seat_specs[0].get_or_insert_with(|| "human".to_string());
            seat_specs[1].get_or_insert_with(|| "human".to_string());
        }
        "4p-teams" => {
            num_players = 4;
            num_teams = 2;
            seat_specs[0].get_or_insert_with(|| "human".to_string());
            seat_specs[1].get_or_insert_with(|| format!("is-mcts:{iters}:5"));
            seat_specs[2].get_or_insert_with(|| format!("macro-heuristic:{iters}:5"));
            seat_specs[3].get_or_insert_with(|| "heuristic".to_string());
        }
        "3p" => {
            num_players = 3;
            num_teams = 3;
            seat_specs[0].get_or_insert_with(|| "human".to_string());
            seat_specs[1].get_or_insert_with(|| format!("is-mcts:{iters}:5"));
            seat_specs[2].get_or_insert_with(|| "heuristic".to_string());
        }
        _ => {
            // human-ai default
            seat_specs[0].get_or_insert_with(|| "human".to_string());
            seat_specs[1].get_or_insert_with(|| format!("is-mcts:{iters}:5"));
        }
    }

    let config = match (num_players, num_teams) {
        (2, _) => SequenceConfig::new_2p(),
        (3, _) => SequenceConfig::new_3p(),
        (4, _) => SequenceConfig::new_4p(),
        (6, t) => SequenceConfig::new_6p(t),
        (n, t) => SequenceConfig {
            num_players: n,
            num_teams: t,
            cards_per_player: 6,
            target_sequences: if t == 3 { 1 } else { 2 },
        },
    };

    println!("============================================================");
    println!("             🎴 SEQUENCE - MULTI-AGENT MCTS 🎴            ");
    println!("============================================================");
    println!(
        "Players: {} | Teams: {} | Target Sequences to Win: {}\n",
        config.num_players, config.num_teams, config.target_sequences
    );

    match num_players {
        2 => run_game::<2>(config, &seat_specs[0..2], use_color),
        3 => run_game::<3>(config, &seat_specs[0..3], use_color),
        4 => run_game::<4>(config, &seat_specs[0..4], use_color),
        6 => run_game::<6>(config, &seat_specs[0..6], use_color),
        _ => run_game::<2>(config, &seat_specs[0..2], use_color),
    }
}

fn run_game<const P: usize>(
    config: SequenceConfig,
    seat_specs: &[Option<String>],
    use_color: bool,
) {
    let mut rng = rand::thread_rng();
    let mut state = SequenceState::new(config, &mut rng);

    let mut agents: Vec<BoxAgent<P>> = (0..P)
        .map(|p| {
            let spec = seat_specs[p]
                .as_deref()
                .unwrap_or(if p == 0 { "human" } else { "heuristic" });
            let name = format!("Player {} ({})", p, TEAM_NAMES[config.player_team(p) as usize]);
            parse_agent::<P>(spec, &name)
        })
        .collect();

    while !state.terminated {
        let active = state.current_player;

        // If active agent is AI, show current state before moving
        let is_human = agents[active].name().contains("Human");
        if !is_human {
            println!("{}", render_state(&state, use_color));
            println!("Thinking... ({} is selecting an action)", agents[active].name());
        }

        let action = agents[active].select_action(&state);
        println!(
            "\n>>> {} chose: {}\n",
            agents[active].name(),
            format_action(&action)
        );

        state.step(&action, &mut rng);
    }

    println!("{}", render_state(&state, use_color));
    match state.winner_team {
        Some(w) => println!(
            "🎉 MATCH OVER! Team {} ({}) emerges victorious in {} total moves!",
            w, TEAM_NAMES[w as usize], state.total_moves
        ),
        None => println!("🤝 MATCH OVER! Game ended in a DRAW in {} total moves!", state.total_moves),
    }
}
