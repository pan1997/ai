//! Benchmarking tournament arena runner for Blokus.
//!
//! Evaluates win rates, piece placements, and average scores with dynamic agent selection
//! and randomized seating to eliminate turn-order and color bias.

use blokus::agent::{Agent, BoxAgent, HeuristicAgent, MctsAgent, RandomAgent};
use blokus::game::{BlokusAction, BlokusState, Player};
use blokus::world::BlokusWorld;
use mcts_engine::arena::{MatchDriver, MultiPlayerTournamentStats, disambiguate_names};
use rand::seq::SliceRandom;
use std::env;
use std::time::Instant;

/// Specification for constructing a Blokus agent.
#[derive(Debug, Clone)]
pub enum AgentSpec {
    /// Monte Carlo Tree Search agent with static area heuristic.
    Mcts {
        /// Number of MCTS simulation sweeps.
        iters: usize,
    },
    /// MCTS agent with heuristic utility priors and static territory values (MCTS-HU).
    MctsHeuristicUtility {
        /// Number of MCTS simulation sweeps.
        iters: usize,
    },
    /// MCTS agent with informed heuristic simulation rollouts (MCTS-HR).
    MctsHeuristicRollout {
        /// Number of MCTS simulation sweeps.
        iters: usize,
        /// Number of rollout simulations per leaf.
        num_rollouts: usize,
        /// Maximum depth per rollout.
        max_depth: usize,
    },
    /// MCTS agent with random rollout simulation playouts.
    MctsRollout {
        /// Number of MCTS simulation sweeps.
        iters: usize,
        /// Number of rollout simulations per leaf.
        num_rollouts: usize,
        /// Maximum depth per rollout.
        max_depth: usize,
    },
    /// MCTS agent with uniform priors and zero values.
    MctsUniform {
        /// Number of MCTS simulation sweeps.
        iters: usize,
    },
    /// Greedy heuristic agent prioritizing pentominoes and corner expansion.
    Heuristic,
    /// Random agent selecting legal moves uniformly at random.
    Random,
}

impl AgentSpec {
    /// Parses an agent specification string.
    pub fn parse(s: &str, default_iters: usize) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts[0].trim().to_lowercase().as_str() {
            "mcts-hu" | "hu" | "mcts-utility" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid iterations in '{s}'"))?
                } else {
                    default_iters
                };
                Ok(Self::MctsHeuristicUtility { iters })
            }
            "mcts-hr" | "hr" | "mcts-heur-rollout" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid iterations in '{s}'"))?
                } else {
                    default_iters
                };
                let num_rollouts = if parts.len() > 2 {
                    parts[2]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid rollouts count in '{s}'"))?
                } else {
                    2
                };
                let max_depth = if parts.len() > 3 {
                    parts[3]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid rollout depth in '{s}'"))?
                } else {
                    20
                };
                Ok(Self::MctsHeuristicRollout {
                    iters,
                    num_rollouts,
                    max_depth,
                })
            }
            "mcts" | "mcts-heuristic" | "heur-mcts" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid iterations in '{s}'"))?
                } else {
                    default_iters
                };
                Ok(Self::Mcts { iters })
            }
            "mcts-rollout" | "rollout" | "mcts-ro" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid iterations in '{s}'"))?
                } else {
                    default_iters
                };
                let num_rollouts = if parts.len() > 2 {
                    parts[2]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid rollouts count in '{s}'"))?
                } else {
                    2
                };
                let max_depth = if parts.len() > 3 {
                    parts[3]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid rollout depth in '{s}'"))?
                } else {
                    20
                };
                Ok(Self::MctsRollout {
                    iters,
                    num_rollouts,
                    max_depth,
                })
            }
            "mcts-uniform" | "uniform" | "mcts-u" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid iterations in '{s}'"))?
                } else {
                    default_iters
                };
                Ok(Self::MctsUniform { iters })
            }
            "heuristic" | "heur" => Ok(Self::Heuristic),
            "random" | "rand" => Ok(Self::Random),
            other => Err(format!(
                "Unknown agent type '{other}'. Expected 'mcts[:iters]', 'mcts-hu[:iters]', 'mcts-hr[:iters[:r[:d]]]', 'mcts-rollout[:iters[:r[:d]]]', 'mcts-uniform[:iters]', 'heuristic', or 'random'"
            )),
        }
    }

    /// Returns the canonical display name.
    pub fn display_name(&self) -> String {
        match self {
            Self::Mcts { iters } => format!("MCTS({iters})"),
            Self::MctsHeuristicUtility { iters } => format!("MCTS-HU({iters})"),
            Self::MctsHeuristicRollout {
                iters,
                num_rollouts,
                max_depth,
            } => format!("MCTS-HR({iters},r={num_rollouts},d={max_depth})"),
            Self::MctsRollout {
                iters,
                num_rollouts,
                max_depth,
            } => format!("MCTS-RO({iters},r={num_rollouts},d={max_depth})"),
            Self::MctsUniform { iters } => format!("MCTS-Uniform({iters})"),
            Self::Heuristic => "Heuristic".to_string(),
            Self::Random => "Random".to_string(),
        }
    }

    /// Instantiates an agent trait object.
    pub fn instantiate<const B: usize, const P: usize>(&self, name: &str) -> BoxAgent<B, P> {
        match self {
            Self::Mcts { iters } => Box::new(MctsAgent::new_heuristic(name, *iters, false)),
            Self::MctsHeuristicUtility { iters } => {
                Box::new(MctsAgent::new_heuristic_utility(name, *iters, false))
            }
            Self::MctsHeuristicRollout {
                iters,
                num_rollouts,
                max_depth,
            } => Box::new(MctsAgent::new_heuristic_rollout(
                name,
                *iters,
                *num_rollouts,
                *max_depth,
                false,
            )),
            Self::MctsRollout {
                iters,
                num_rollouts,
                max_depth,
            } => Box::new(MctsAgent::new_rollout(
                name,
                *iters,
                *num_rollouts,
                *max_depth,
                false,
            )),
            Self::MctsUniform { iters } => Box::new(MctsAgent::new_uniform(name, *iters, false)),
            Self::Heuristic => Box::new(HeuristicAgent::new(name)),
            Self::Random => Box::new(RandomAgent::new(name)),
        }
    }
}

/// Disambiguates duplicate agent names by appending sequential numeric suffixes.
fn generate_unique_names(specs: &[AgentSpec]) -> Vec<String> {
    let raw_names: Vec<String> = specs.iter().map(|s| s.display_name()).collect();
    disambiguate_names(&raw_names)
}

fn print_help() {
    println!(
        r#"Blokus Tournament Arena

USAGE:
    blokus-tournament [OPTIONS]

OPTIONS:
    --agents <SPECS> (or --players)
                       Comma-separated list of agents to compete.
                       Supported agent specs:
                         'mcts:<iters>'                            MCTS with static area heuristic
                         'mcts-hu:<iters>'                         MCTS with Heuristic Utility priors (MCTS-HU)
                         'mcts-hr:<iters>:<rollouts>:<depth>'      MCTS with Heuristic simulation rollouts (MCTS-HR)
                         'mcts-rollout:<iters>:<rollouts>:<depth>' MCTS with random simulation rollouts
                         'mcts-uniform:<iters>'                    MCTS with uniform baseline priors & values
                         'heuristic' (or 'heur')                   Greedy 1-ply corner/pentomino heuristic
                         'random' (or 'rand')                      Uniform random legal moves
                       Examples:
                         --agents heuristic,mcts-hu:1000,mcts-hr:500:2:15,mcts:1000   (4-player Classic)
                         --agents mcts-hu:1000,heuristic                             (2-player Duo)

    --agent <SPEC> (or --player)
                       Repeatable flag to add an individual agent.
                       Example: --agent mcts-hu:1000 --agent mcts-hr:500:2 --agent heuristic --agent random

    --board-size <N> (or --size, --mode)
                       Explicit variant: '14'/'duo' (14x14, 2P) or '20'/'classic' (20x20, 4P).
                       Inferred automatically if 2 or 4 players are given.

    --games <N> (or --rounds)
                       Number of tournament games to simulate [default: 10]
    --iters <N>        Default MCTS iterations if omitted from spec [default: 500]
    -h, --help         Print help information
"#
    );
}

fn run_tournament<const B: usize, const P: usize>(
    specs: &[AgentSpec],
    names: &[String],
    num_games: usize,
    variant_name: &str,
) {
    println!(
        "=========================================================================================="
    );
    println!(
        "            ⚔️  BLOKUS {variant_name} TOURNAMENT (RANDOMIZED SEATING)  ⚔️             "
    );
    println!(
        "=========================================================================================="
    );
    println!(
        "Games: {num_games} | Players ({P}): {}\n",
        names.join(" vs ")
    );

    let mut stats: Vec<MultiPlayerTournamentStats> =
        (0..P).map(|_| MultiPlayerTournamentStats::new(P)).collect();
    let mut rng = rand::thread_rng();
    let start_time = Instant::now();

    for game_idx in 1..=num_games {
        // Randomly shuffle seats to eliminate turn order / color advantage
        let mut seat_to_agent: Vec<usize> = (0..P).collect();
        seat_to_agent.shuffle(&mut rng);

        // Instantiate fresh agent instances for each seat
        let mut active_players: Vec<BoxAgent<B, P>> = seat_to_agent
            .iter()
            .map(|&agent_idx| specs[agent_idx].instantiate(&names[agent_idx]))
            .collect();

        let world = BlokusWorld::<B, P>::new();
        let driver = MatchDriver::new();
        let mut agent_refs: Vec<&mut dyn Agent<BlokusState<B, P>, BlokusAction>> = active_players
            .iter_mut()
            .map(|a| &mut **a as &mut dyn Agent<BlokusState<B, P>, BlokusAction>)
            .collect();

        let match_result = driver.play_multi::<_, _, P>(&world, &mut agent_refs, None);
        let state = match_result.final_state;

        // Compute scores and record match statistics
        let scores: Vec<i32> = (0..P).map(|s| state.score(s)).collect();
        let placed: Vec<f64> = (0..P)
            .map(|s| (89 - (state.unplaced_squares(s) as usize)) as f64)
            .collect();
        MultiPlayerTournamentStats::record_game(&mut stats, &seat_to_agent, &scores, Some(&placed));

        // Format per-game breakdown
        let mut seat_details = Vec::new();
        for seat in 0..P {
            let agent_idx = seat_to_agent[seat];
            let color_name = Player::from_index(seat).name();
            seat_details.push(format!(
                "{}: {} ({})",
                color_name, names[agent_idx], scores[seat]
            ));
        }

        let max_score = *scores.iter().max().unwrap();
        let winner_names: Vec<&str> = (0..P)
            .filter(|&s| scores[s] == max_score)
            .map(|s| names[seat_to_agent[s]].as_str())
            .collect();

        println!(
            "Game {:3}/{num_games}: [{}] => Winner(s): {:?}",
            game_idx,
            seat_details.join(", "),
            winner_names
        );
    }

    let elapsed = start_time.elapsed();
    let seat_names: Vec<&str> = (0..P).map(|s| Player::from_index(s).name()).collect();
    let rank_indices = MultiPlayerTournamentStats::print_leaderboard(
        &stats,
        names,
        num_games,
        elapsed,
        Some("Placed / 89"),
    );
    MultiPlayerTournamentStats::print_seating_fairness(&stats, names, &rank_indices, &seat_names);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut mode_arg: Option<String> = None;
    let mut games = 10;
    let mut default_iters = 500;
    let mut player_specs: Vec<AgentSpec> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" | "--board-size" | "--size" => {
                if i + 1 < args.len() {
                    let val = args[i + 1].to_lowercase();
                    mode_arg = match val.as_str() {
                        "14" | "duo" => Some("duo".to_string()),
                        "20" | "classic" => Some("classic".to_string()),
                        _ => Some(val),
                    };
                    i += 1;
                }
            }
            "--games" | "--rounds" => {
                if i + 1 < args.len() {
                    games = args[i + 1].parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--iters" => {
                if i + 1 < args.len() {
                    default_iters = args[i + 1].parse().unwrap_or(500);
                    i += 1;
                }
            }
            "--players" | "--agents" => {
                if i + 1 < args.len() {
                    let raw = &args[i + 1];
                    for item in raw.split(',') {
                        let trimmed = item.trim();
                        if !trimmed.is_empty() {
                            match AgentSpec::parse(trimmed, default_iters) {
                                Ok(spec) => player_specs.push(spec),
                                Err(err) => {
                                    eprintln!("Error parsing agent spec: {err}");
                                    std::process::exit(1);
                                }
                            }
                        }
                    }
                    i += 1;
                }
            }
            "--player" | "--agent" => {
                if i + 1 < args.len() {
                    let raw = &args[i + 1];
                    match AgentSpec::parse(raw, default_iters) {
                        Ok(spec) => player_specs.push(spec),
                        Err(err) => {
                            eprintln!("Error parsing agent spec: {err}");
                            std::process::exit(1);
                        }
                    }
                    i += 1;
                }
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    // If no players explicitly passed, select defaults based on mode
    if player_specs.is_empty() {
        let is_duo = matches!(mode_arg.as_deref(), Some("duo"));
        if is_duo {
            player_specs.push(AgentSpec::Mcts {
                iters: default_iters,
            });
            player_specs.push(AgentSpec::Heuristic);
        } else {
            player_specs.push(AgentSpec::Mcts {
                iters: default_iters,
            });
            player_specs.push(AgentSpec::Heuristic);
            player_specs.push(AgentSpec::Random);
            player_specs.push(AgentSpec::Random);
        }
    }

    let names = generate_unique_names(&player_specs);

    match player_specs.len() {
        2 => {
            if let Some(ref m) = mode_arg
                && m == "classic"
            {
                eprintln!(
                    "Error: 2 players were specified, but mode 'classic' requires 4 players."
                );
                std::process::exit(1);
            }
            run_tournament::<14, 2>(&player_specs, &names, games, "DUO (14x14)");
        }
        4 => {
            if let Some(ref m) = mode_arg
                && m == "duo"
            {
                eprintln!("Error: 4 players were specified, but mode 'duo' requires 2 players.");
                std::process::exit(1);
            }
            run_tournament::<20, 4>(&player_specs, &names, games, "CLASSIC (20x20)");
        }
        count => {
            eprintln!(
                "Error: Tournament currently supports either 2 players (Duo) or 4 players (Classic), but {count} players were provided."
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_blokus_specs() {
        assert!(matches!(
            AgentSpec::parse("random", 100).unwrap(),
            AgentSpec::Random
        ));
        assert!(matches!(
            AgentSpec::parse("heuristic", 100).unwrap(),
            AgentSpec::Heuristic
        ));
        assert!(matches!(
            AgentSpec::parse("mcts:50", 100).unwrap(),
            AgentSpec::Mcts { iters: 50 }
        ));
        assert!(matches!(
            AgentSpec::parse("mcts-hu:80", 100).unwrap(),
            AgentSpec::MctsHeuristicUtility { iters: 80 }
        ));
        assert!(matches!(
            AgentSpec::parse("mcts-hr:100:2:15", 100).unwrap(),
            AgentSpec::MctsHeuristicRollout {
                iters: 100,
                num_rollouts: 2,
                max_depth: 15
            }
        ));
        assert!(matches!(
            AgentSpec::parse("mcts-rollout:120:3:20", 100).unwrap(),
            AgentSpec::MctsRollout {
                iters: 120,
                num_rollouts: 3,
                max_depth: 20
            }
        ));
        assert!(matches!(
            AgentSpec::parse("mcts-u:40", 100).unwrap(),
            AgentSpec::MctsUniform { iters: 40 }
        ));
    }

    #[test]
    fn test_generate_unique_names() {
        let specs = vec![AgentSpec::Random, AgentSpec::Random, AgentSpec::Heuristic];
        let names = generate_unique_names(&specs);
        assert_eq!(names, vec!["Random-1", "Random-2", "Heuristic"]);
    }
}
