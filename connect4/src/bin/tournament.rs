//! Benchmarking tournament arena runner for Connect 4.
//!
//! Evaluates win rates, tactical efficacy, and throughput across arbitrary lists of agents ($K \ge 2$).
//! When $K \ge 2$ agents are provided, runs a complete round-robin tournament where every pair of agents
//! plays the specified number of games with strictly balanced first-mover advantage (half Red, half Yellow).
//!
//! # Usage Examples
//! ```bash
//! # 1. Multi-agent round-robin tournament across 4 diverse agent types
//! cargo run -p connect4 --bin connect4-tournament -- --agents round-adversarial:100,mcts:100,macro-tactical:100,tactical --games 10
//!
//! # 2. Using repeatable --agent flags
//! cargo run -p connect4 --bin connect4-tournament -- --agent round-adversarial:200 --agent mcts:200 --agent tactical --games 20
//!
//! # 3. Head-to-head match between two specific agents
//! cargo run -p connect4 --bin connect4-tournament -- --p1 round-adversarial:200 --p2 tactical --games 20
//! ```

use connect4::agent::{Agent, Connect4AgentSpec, generate_unique_names};
use connect4::game::{Connect4State, Player};
use mcts_engine::arena::{GameOutcome, H2HMatrix, TwoPlayerTournamentStats};
use std::env;
use std::time::Instant;

fn print_help() {
    println!(
        r#"Connect 4 - Round-Robin Tournament Arena

USAGE:
    connect4-tournament [OPTIONS]

OPTIONS:
    --agents <SPECS> (or --players)
                             Comma-separated list of agent specifications.
                             Examples:
                               --agents mcts:2000,tactical,random
                               --agents mcts:2000,macro-tactical:2000,macro-random:2000
    --agent <SPEC> (or --player)
                             Repeatable flag to add an individual agent.
                             Example: --agent mcts:2000 --agent tactical --agent random
    --games <N>              Games to play per pairwise matchup [default: 10]
                             (half played as Red, half played as Yellow)
    --iters <N>              Default MCTS iterations when omitted from spec [default: 200]
    --rollouts <N>           Default rollout evaluations per leaf [default: 3] (0 for uniform)
    --c-puct <FLOAT>         PUCT exploration constant [default: 1.414]
    --verbose                Print search candidate tables for each move
    -h, --help               Print help information

PAIRWISE MATCHUP FLAGS (2-player shorthand):
    --p1 <SPEC>              Player 1 agent spec [default: mcts]
    --p1-iters <N>           MCTS iterations for Player 1
    --p1-rollouts <N>        Rollouts per leaf for Player 1
    --p2 <SPEC>              Player 2 agent spec [default: random]
    --p2-iters <N>           MCTS iterations for Player 2
    --p2-rollouts <N>        Rollouts per leaf for Player 2

SUPPORTED AGENT SPECS:
    mcts[:iters[:rollouts]]            Standard zero-sum adversarial MCTS (default)
    macro-tactical[:iters[:rollouts]]  Macro MCTS assuming opponent plays tactical heuristic
    macro-random[:iters[:rollouts]]    Macro MCTS assuming opponent plays uniformly random
    tactical (or heur)                 Greedy tactical heuristic (immediate win, block 1-ply win)
    random (or rand)                   Uniform random legal moves

BACKWARD-COMPATIBILITY ALIASES:
    round-adversarial                  Alias for mcts
    round-tactical                     Alias for macro-tactical
    round-random                       Alias for macro-random
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut games_per_pair = 10;
    let mut default_iters = 200;
    let mut default_rollouts = 3;
    let mut c_puct = 1.414;
    let mut verbose = false;

    let mut agent_specs: Vec<Connect4AgentSpec> = Vec::new();

    // Pairwise override options
    let mut p1_opt: Option<String> = None;
    let mut p1_iters_opt: Option<usize> = None;
    let mut p1_rollouts_opt: Option<usize> = None;

    let mut p2_opt: Option<String> = None;
    let mut p2_iters_opt: Option<usize> = None;
    let mut p2_rollouts_opt: Option<usize> = None;

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
                    games_per_pair = args[i].parse().unwrap_or(10);
                }
            }
            "--iters" => {
                i += 1;
                if i < args.len() {
                    default_iters = args[i].parse().unwrap_or(200);
                }
            }
            "--rollouts" => {
                i += 1;
                if i < args.len() {
                    default_rollouts = args[i].parse().unwrap_or(3);
                }
            }
            "--c-puct" => {
                i += 1;
                if i < args.len() {
                    c_puct = args[i].parse().unwrap_or(1.414);
                }
            }
            "--verbose" => {
                verbose = true;
            }
            "--agents" | "--players" => {
                i += 1;
                if i < args.len() {
                    for item in args[i].split(',') {
                        let trimmed = item.trim();
                        if !trimmed.is_empty() {
                            match Connect4AgentSpec::parse(trimmed, default_iters, default_rollouts)
                            {
                                Ok(spec) => agent_specs.push(spec),
                                Err(err) => {
                                    eprintln!("Error: {err}");
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            "--agent" | "--player" => {
                i += 1;
                if i < args.len() {
                    let trimmed = args[i].trim();
                    match Connect4AgentSpec::parse(trimmed, default_iters, default_rollouts) {
                        Ok(spec) => agent_specs.push(spec),
                        Err(err) => {
                            eprintln!("Error: {err}");
                            return;
                        }
                    }
                }
            }
            "--p1" | "--agent1" => {
                i += 1;
                if i < args.len() {
                    p1_opt = Some(args[i].clone());
                }
            }
            "--p1-iters" | "--mcts-iters" => {
                i += 1;
                if i < args.len() {
                    p1_iters_opt = Some(args[i].parse().unwrap_or(200));
                }
            }
            "--p1-rollouts" => {
                i += 1;
                if i < args.len() {
                    p1_rollouts_opt = Some(args[i].parse().unwrap_or(3));
                }
            }
            "--p2" | "--agent2" | "--opponent" => {
                i += 1;
                if i < args.len() {
                    p2_opt = Some(args[i].clone());
                }
            }
            "--p2-iters" | "--opponent-iters" => {
                i += 1;
                if i < args.len() {
                    p2_iters_opt = Some(args[i].parse().unwrap_or(100));
                }
            }
            "--p2-rollouts" => {
                i += 1;
                if i < args.len() {
                    p2_rollouts_opt = Some(args[i].parse().unwrap_or(3));
                }
            }
            other => {
                eprintln!("Unknown option '{other}'. Run with --help for usage.");
                return;
            }
        }
        i += 1;
    }

    // If agent_specs was not populated via --agents/--agent, check --p1 / --p2 or default pair
    if agent_specs.is_empty() {
        let p1_str = p1_opt.unwrap_or_else(|| "mcts".to_string());
        let p1_iters = p1_iters_opt.unwrap_or(default_iters);
        let p1_rollouts = p1_rollouts_opt.unwrap_or(default_rollouts);
        let p1_spec = match Connect4AgentSpec::parse(&p1_str, p1_iters, p1_rollouts) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error in P1: {e}");
                return;
            }
        };

        let p2_str = p2_opt.unwrap_or_else(|| "random".to_string());
        let p2_iters = p2_iters_opt.unwrap_or(default_iters);
        let p2_rollouts = p2_rollouts_opt.unwrap_or(default_rollouts);
        let p2_spec = match Connect4AgentSpec::parse(&p2_str, p2_iters, p2_rollouts) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error in P2: {e}");
                return;
            }
        };

        agent_specs.push(p1_spec);
        agent_specs.push(p2_spec);
    }

    let num_agents = agent_specs.len();
    if num_agents < 2 {
        eprintln!("Error: At least 2 agents are required to run a tournament.");
        return;
    }

    let unique_names = generate_unique_names(&agent_specs);
    let num_pairs = num_agents * (num_agents - 1) / 2;
    let total_tournament_games = num_pairs * games_per_pair;

    println!(
        "=========================================================================================="
    );
    println!(
        "                         ⚔️  CONNECT 4 TOURNAMENT ARENA ⚔️                               "
    );
    println!(
        "=========================================================================================="
    );
    println!("Agents competing ({num_agents}):");
    for (idx, name) in unique_names.iter().enumerate() {
        println!("  [{}] {name}", idx + 1);
    }
    println!(
        "\nFormat:       Round-Robin (every pair plays {games_per_pair} games: half Red, half Yellow)"
    );
    println!("Matchups:     {num_pairs} distinct pairings");
    println!("Total Games:  {total_tournament_games}");
    println!("C_PUCT:       {c_puct}\n");

    let mut stats = vec![TwoPlayerTournamentStats::default(); num_agents];
    let mut h2h = H2HMatrix::new(num_agents);

    let start_time = Instant::now();
    let mut game_counter = 0;

    for i in 0..num_agents {
        for j in (i + 1)..num_agents {
            let name_i = &unique_names[i];
            let name_j = &unique_names[j];

            println!(
                "------------------------------------------------------------------------------------------"
            );
            println!("▶ Matchup: {name_i} vs. {name_j} ({games_per_pair} games)");
            println!(
                "------------------------------------------------------------------------------------------"
            );

            let mut m_wins_i = 0;
            let mut m_wins_j = 0;
            let mut m_draws = 0;

            for game_idx in 0..games_per_pair {
                game_counter += 1;
                // Strictly balanced coloring: alternate who plays Red
                let i_is_red = game_idx % 2 == 0;
                let (red_idx, yellow_idx) = if i_is_red { (i, j) } else { (j, i) };

                let red_name = format!("{} (Red)", unique_names[red_idx]);
                let yellow_name = format!("{} (Yellow)", unique_names[yellow_idx]);

                let mut red_agent = agent_specs[red_idx].instantiate(&red_name, c_puct, verbose);
                let mut yellow_agent =
                    agent_specs[yellow_idx].instantiate(&yellow_name, c_puct, verbose);

                let mut state = Connect4State::<6, 7>::new();
                let mut moves: usize = 0;

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

                let outcome = match game_result {
                    Some(Player::Red) => GameOutcome::Seat0Wins,
                    Some(Player::Yellow) => GameOutcome::Seat1Wins,
                    None => GameOutcome::Draw,
                };

                h2h.record_game(red_idx, yellow_idx, outcome);
                TwoPlayerTournamentStats::record_game(
                    &mut stats,
                    red_idx,
                    yellow_idx,
                    outcome,
                    moves.div_ceil(2),
                    moves / 2,
                );

                match outcome {
                    GameOutcome::Seat0Wins => {
                        if red_idx == i {
                            m_wins_i += 1;
                        } else {
                            m_wins_j += 1;
                        }
                        println!(
                            "  Game #{:>3}: {} (Red) won in {moves} moves",
                            game_counter, unique_names[red_idx]
                        );
                    }
                    GameOutcome::Seat1Wins => {
                        if yellow_idx == i {
                            m_wins_i += 1;
                        } else {
                            m_wins_j += 1;
                        }
                        println!(
                            "  Game #{:>3}: {} (Yellow) won in {moves} moves",
                            game_counter, unique_names[yellow_idx]
                        );
                    }
                    GameOutcome::Draw => {
                        m_draws += 1;
                        println!("  Game #{:>3}: Draw in {moves} moves", game_counter);
                    }
                }
            }

            println!(
                "  ↳ Matchup Outcome: {} {}-{}-{} {}",
                name_i, m_wins_i, m_wins_j, m_draws, name_j
            );
        }
    }

    let elapsed = start_time.elapsed();
    let total_all_moves: usize = stats.iter().map(|s| s.total_moves).sum();
    let moves_per_sec = total_all_moves as f64 / elapsed.as_secs_f64().max(0.001);

    // Print Head-to-Head Cross Table
    h2h.print_table(&unique_names);

    // Print Final Leaderboard Standings sorted by Win Rate
    TwoPlayerTournamentStats::print_standings(&stats, &unique_names, "Red", "Yel");

    println!("Total Matchups:   {num_pairs}");
    println!("Total Games:      {total_tournament_games}");
    println!("Elapsed Time:     {:.2?}", elapsed);
    println!("Total Moves:      {total_all_moves}");
    println!("Throughput:       {:.1} moves/sec", moves_per_sec);
    println!(
        "=========================================================================================="
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_all_agent_specs() {
        assert_eq!(
            Connect4AgentSpec::parse("random", 200, 3).unwrap(),
            Connect4AgentSpec::Random
        );
        assert_eq!(
            Connect4AgentSpec::parse("tactical", 200, 3).unwrap(),
            Connect4AgentSpec::Tactical
        );
        assert_eq!(
            Connect4AgentSpec::parse("mcts", 200, 3).unwrap(),
            Connect4AgentSpec::Mcts {
                iters: 200,
                rollouts: 3
            }
        );
        assert_eq!(
            Connect4AgentSpec::parse("mcts:500:5", 200, 3).unwrap(),
            Connect4AgentSpec::Mcts {
                iters: 500,
                rollouts: 5
            }
        );
        assert_eq!(
            Connect4AgentSpec::parse("round-adversarial:100:0", 200, 3).unwrap(),
            Connect4AgentSpec::Mcts {
                iters: 100,
                rollouts: 0
            }
        );
        assert_eq!(
            Connect4AgentSpec::parse("round-tactical:150:2", 200, 3).unwrap(),
            Connect4AgentSpec::MacroTactical {
                iters: 150,
                rollouts: 2
            }
        );
        assert_eq!(
            Connect4AgentSpec::parse("macro-tactical:80", 200, 3).unwrap(),
            Connect4AgentSpec::MacroTactical {
                iters: 80,
                rollouts: 3
            }
        );
    }

    #[test]
    fn test_unique_name_generation() {
        let specs = vec![
            Connect4AgentSpec::Random,
            Connect4AgentSpec::Random,
            Connect4AgentSpec::Tactical,
        ];
        let names = generate_unique_names(&specs);
        assert_eq!(names, vec!["Random-1", "Random-2", "Tactical"]);
    }

    #[test]
    fn test_instantiate_all_specs() {
        let specs = [
            Connect4AgentSpec::Random,
            Connect4AgentSpec::Tactical,
            Connect4AgentSpec::Mcts {
                iters: 10,
                rollouts: 0,
            },
            Connect4AgentSpec::Mcts {
                iters: 10,
                rollouts: 1,
            },
            Connect4AgentSpec::MacroTactical {
                iters: 10,
                rollouts: 0,
            },
            Connect4AgentSpec::MacroRandom {
                iters: 10,
                rollouts: 0,
            },
        ];

        let state = Connect4State::<6, 7>::new();
        for spec in &specs {
            let mut agent = spec.instantiate("Test", 1.414, false);
            let action = agent.select_action(&state);
            assert!(action < 7);
        }
    }
}
