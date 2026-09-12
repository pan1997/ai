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

use connect4::agent::{Agent, MctsAgent, RandomAgent, TacticalAgent};
use connect4::evaluator::{RolloutEvaluator, UniformEvaluator};
use connect4::game::{Connect4State, Player};
use mcts_engine::arena::{GameOutcome, H2HMatrix, TwoPlayerTournamentStats, disambiguate_names};
use std::env;
use std::time::Instant;

/// Specification for constructing a Connect 4 agent from CLI arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Connect4AgentSpec {
    /// Baseline agent playing uniform random legal moves.
    Random,
    /// Greedy tactical heuristic (immediate win, block 1-ply win, center preference).
    Tactical,
    /// Standard 1-ply alternating MCTS (SequentialScheduler + TurnBasedDynamics).
    Mcts { iters: usize, rollouts: usize },
    /// Coordinated round-based MCTS with AdversarialOpponent in the same tree (RoundScheduler).
    RoundAdversarial { iters: usize, rollouts: usize },
    /// Coordinated round-based MCTS with TacticalOpponent in tree (RoundScheduler).
    RoundTactical { iters: usize, rollouts: usize },
    /// Coordinated round-based MCTS with RandomOpponent in tree (RoundScheduler).
    RoundRandom { iters: usize, rollouts: usize },
    /// Absorbed macro-dynamics MCTS with TacticalOpponent (Route B).
    MacroTactical { iters: usize, rollouts: usize },
    /// Absorbed macro-dynamics MCTS with RandomOpponent (Route B).
    MacroRandom { iters: usize, rollouts: usize },
}

impl Connect4AgentSpec {
    /// Parses an agent specification string (e.g. `round-adversarial:200:3`, `mcts:100`, `tactical`, `random`).
    pub fn parse(s: &str, default_iters: usize, default_rollouts: usize) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        let parse_iters = |idx: usize| -> Result<usize, String> {
            if parts.len() > idx {
                parts[idx]
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid iteration count in '{s}'"))
            } else {
                Ok(default_iters)
            }
        };

        let parse_rollouts = |idx: usize| -> Result<usize, String> {
            if parts.len() > idx {
                parts[idx]
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid rollouts count in '{s}'"))
            } else {
                Ok(default_rollouts)
            }
        };

        match parts[0].trim().to_lowercase().as_str() {
            "random" | "rand" => Ok(Self::Random),
            "tactical" | "heur" | "heuristic" => Ok(Self::Tactical),
            "mcts" | "mcts-sequential" | "sequential" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::Mcts { iters, rollouts })
            }
            "round-adversarial" | "round-mcts-adversarial" | "round-adv" | "round" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::RoundAdversarial { iters, rollouts })
            }
            "round-tactical" | "round-mcts-tactical" | "round-tact" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::RoundTactical { iters, rollouts })
            }
            "round-random" | "round-mcts-random" | "round-rand" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::RoundRandom { iters, rollouts })
            }
            "macro-tactical" | "macro-mcts-tactical" | "macro-tact" | "macro" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::MacroTactical { iters, rollouts })
            }
            "macro-random" | "macro-mcts-random" | "macro-rand" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::MacroRandom { iters, rollouts })
            }
            other => Err(format!(
                "Unknown agent type '{other}'. Supported: random, tactical, mcts[:iters[:rollouts]], round-adversarial[:iters[:rollouts]], round-tactical[:iters[:rollouts]], round-random[:iters[:rollouts]], macro-tactical[:iters[:rollouts]], macro-random[:iters[:rollouts]]"
            )),
        }
    }

    /// Returns a human-readable display name summarizing type and hyperparameters.
    pub fn display_name(&self) -> String {
        match self {
            Self::Random => "Random".to_string(),
            Self::Tactical => "Tactical".to_string(),
            Self::Mcts { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("MCTS({iters},{eval})")
            }
            Self::RoundAdversarial { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Round-Adv({iters},{eval})")
            }
            Self::RoundTactical { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Round-Tact({iters},{eval})")
            }
            Self::RoundRandom { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Round-Rand({iters},{eval})")
            }
            Self::MacroTactical { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Macro-Tact({iters},{eval})")
            }
            Self::MacroRandom { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Macro-Rand({iters},{eval})")
            }
        }
    }

    /// Instantiates an agent trait object ready for game execution.
    pub fn instantiate(
        &self,
        name: &str,
        c_puct: f32,
        verbose: bool,
    ) -> connect4::agent::BoxAgent<6, 7> {
        match self {
            Self::Random => Box::new(RandomAgent::new(name)),
            Self::Tactical => Box::new(TacticalAgent::new(name)),
            Self::Mcts { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_with_model(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_rollout(name, *iters, *rollouts, 20, verbose))
                }
            }
            Self::RoundAdversarial { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_adversarial(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_adversarial(
                        name,
                        *iters,
                        c_puct,
                        RolloutEvaluator::new(*rollouts, 20),
                        verbose,
                    ))
                }
            }
            Self::RoundTactical { iters, rollouts } | Self::MacroTactical { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_macro_tactical(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_macro_tactical(
                        name,
                        *iters,
                        c_puct,
                        RolloutEvaluator::new(*rollouts, 20),
                        verbose,
                    ))
                }
            }
            Self::RoundRandom { iters, rollouts } | Self::MacroRandom { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_macro_random(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_macro_random(
                        name,
                        *iters,
                        c_puct,
                        RolloutEvaluator::new(*rollouts, 20),
                        verbose,
                    ))
                }
            }
        }
    }
}

/// Disambiguates duplicate agent names by appending sequential numeric suffixes.
fn generate_unique_names(specs: &[Connect4AgentSpec]) -> Vec<String> {
    let raw_names: Vec<String> = specs.iter().map(|s| s.display_name()).collect();
    disambiguate_names(&raw_names)
}

fn print_help() {
    println!(
        r#"Connect 4 - Round-Robin Tournament Arena

USAGE:
    connect4-tournament [OPTIONS]

OPTIONS:
    --agents <SPECS>         Comma-separated list of agent specifications.
                             Examples:
                               --agents round-adversarial:200,mcts:200,tactical,random
                               --agents round-tactical:100:0,macro-tactical:100:0
    --agent <SPEC>           Repeatable flag to add an individual agent to the tournament.
                             Example: --agent round-adversarial:200 --agent mcts:200 --agent tactical
    --games <N>              Games to play per pairwise matchup [default: 10]
                             (half played as Red, half played as Yellow)
    --iters <N>              Default MCTS iterations when omitted from spec [default: 200]
    --rollouts <N>           Default rollout evaluations per leaf [default: 3] (0 for uniform)
    --c-puct <FLOAT>         PUCT exploration constant [default: 1.414]
    --verbose                Print search candidate tables for each move
    -h, --help               Print help information

PAIRWISE MATCHUP FLAGS (2-player shorthand):
    --p1 <SPEC>              Player 1 agent spec [default: round-adversarial]
    --p1-iters <N>           MCTS iterations for Player 1
    --p1-rollouts <N>        Rollouts per leaf for Player 1
    --p2 <SPEC>              Player 2 agent spec [default: random]
    --p2-iters <N>           MCTS iterations for Player 2
    --p2-rollouts <N>        Rollouts per leaf for Player 2

BACKWARD-COMPATIBILITY ALIASES:
    --players <SPECS>        Alias for --agents
    --player <SPEC>          Alias for --agent
    --mcts-iters <N>         Alias for --p1-iters
    --opponent <TYPE>        Alias for --p2
    --opponent-iters <N>     Alias for --p2-iters

SUPPORTED AGENT TYPES:
    random                   Uniform random legal moves
    tactical (or heur)       Greedy tactical heuristic (immediate win, block 1-ply win, center)
    mcts[:iters[:rollouts]]  Standard 1-ply alternating MCTS (SequentialScheduler)
    round-adversarial[:...]  Coordinated round-based MCTS with AdversarialOpponent (RoundScheduler)
    round-tactical[:...]     Coordinated round-based MCTS with TacticalOpponent (RoundScheduler)
    round-random[:...]       Coordinated round-based MCTS with RandomOpponent (RoundScheduler)
    macro-tactical[:...]     Absorbed macro-dynamics MCTS with TacticalOpponent (Route B)
    macro-random[:...]       Absorbed macro-dynamics MCTS with RandomOpponent (Route B)
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
        let p1_str = p1_opt.unwrap_or_else(|| "round-adversarial".to_string());
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
            Connect4AgentSpec::RoundAdversarial {
                iters: 100,
                rollouts: 0
            }
        );
        assert_eq!(
            Connect4AgentSpec::parse("round-tactical:150:2", 200, 3).unwrap(),
            Connect4AgentSpec::RoundTactical {
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
            Connect4AgentSpec::RoundAdversarial {
                iters: 10,
                rollouts: 1,
            },
            Connect4AgentSpec::RoundTactical {
                iters: 10,
                rollouts: 0,
            },
            Connect4AgentSpec::RoundRandom {
                iters: 10,
                rollouts: 0,
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
