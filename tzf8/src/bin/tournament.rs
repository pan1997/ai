//! Standardized benchmark tournament arena evaluating 2048 agents across identical seeded boards.

use mcts_traits::Agent;
use std::env;
use std::time::Instant;
use tzf8::agent::Tzf8AgentSpec;
use tzf8::game::Tzf8State;
use tzf8::world::Tzf8World;

#[derive(Debug)]
struct AgentStats {
    name: String,
    scores: Vec<u32>,
    max_tiles: Vec<u32>,
    moves_counts: Vec<usize>,
    total_duration_secs: f64,
}

impl AgentStats {
    fn new(name: String) -> Self {
        Self {
            name,
            scores: Vec::new(),
            max_tiles: Vec::new(),
            moves_counts: Vec::new(),
            total_duration_secs: 0.0,
        }
    }

    fn avg_score(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.scores.iter().map(|&s| s as u64).sum();
        sum as f64 / self.scores.len() as f64
    }

    fn max_score(&self) -> u32 {
        self.scores.iter().copied().max().unwrap_or(0)
    }

    fn avg_moves(&self) -> f64 {
        if self.moves_counts.is_empty() {
            return 0.0;
        }
        let sum: usize = self.moves_counts.iter().sum();
        sum as f64 / self.moves_counts.len() as f64
    }

    fn total_moves(&self) -> usize {
        self.moves_counts.iter().sum()
    }

    fn pct_tile(&self, threshold: u32) -> f64 {
        if self.max_tiles.is_empty() {
            return 0.0;
        }
        let count = self.max_tiles.iter().filter(|&&t| t >= threshold).count();
        count as f64 * 100.0 / self.max_tiles.len() as f64
    }

    fn moves_per_sec(&self) -> f64 {
        if self.total_duration_secs <= 0.0 {
            return 0.0;
        }
        self.total_moves() as f64 / self.total_duration_secs
    }
}

fn print_usage() {
    println!("2048 Evaluation Arena - Compare agents across standardized identical boards");
    println!();
    println!("Usage: cargo run --release -p tzf8 --bin tzf8-tournament -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --agents <specs>   Comma-separated list of agent specifications");
    println!("                     Examples: 'heuristic,mcts:200,random', 'mcts-rollout:100:5'");
    println!("                     (default: 'heuristic,mcts:100,random')");
    println!(
        "  --boards <N>       Number of standardized boards to evaluate per agent (default: 10)"
    );
    println!("  --seed <S>         Base random seed for board reproducibility (default: 42)");
    println!("  --verbose          Print per-board match progress");
    println!("  --help             Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut agent_specs_str = "heuristic,mcts:100,random".to_string();
    let mut num_boards: usize = 10;
    let mut base_seed: u64 = 42;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agents" => {
                if i + 1 < args.len() {
                    agent_specs_str = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --agents requires a value");
                    std::process::exit(1);
                }
            }
            "--boards" => {
                if i + 1 < args.len() {
                    num_boards = args[i + 1].parse().expect("Invalid --boards number");
                    i += 2;
                } else {
                    eprintln!("Error: --boards requires a value");
                    std::process::exit(1);
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    base_seed = args[i + 1].parse().expect("Invalid --seed number");
                    i += 2;
                } else {
                    eprintln!("Error: --seed requires a value");
                    std::process::exit(1);
                }
            }
            "--verbose" => {
                verbose = true;
                i += 1;
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

    let specs: Vec<Tzf8AgentSpec> = agent_specs_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| Tzf8AgentSpec::parse(s).unwrap_or_else(|e| panic!("Error parsing '{s}': {e}")))
        .collect();

    if specs.is_empty() {
        eprintln!("Error: No agents specified.");
        std::process::exit(1);
    }

    println!(
        "=========================================================================================="
    );
    println!(
        "                                   2048 TOURNAMENT ARENA                                  "
    );
    println!(
        "=========================================================================================="
    );
    println!(
        "Boards: {} | Base Seed: {} | Agents: {}",
        num_boards,
        base_seed,
        specs.len()
    );
    println!(
        "------------------------------------------------------------------------------------------"
    );

    let arena_start = Instant::now();
    let mut all_stats: Vec<AgentStats> = Vec::new();

    for spec in &specs {
        let mut agent = spec.build_agent();
        let agent_name = agent.name().to_string();
        println!(">>> Evaluating Agent: {:<25}", agent_name);

        let mut stats = AgentStats::new(agent_name);
        let agent_start = Instant::now();

        for b in 0..num_boards {
            let board_seed = base_seed.wrapping_add((b as u64).wrapping_mul(10007));
            let world = Tzf8World::with_seed(board_seed);
            let mut state: Tzf8State = Tzf8World::initial_with_seed(board_seed);

            let mut moves = 0;
            while !world.is_terminal(&state) {
                let action = agent.select_action(&state);
                let outcome = world.step_action(&mut state, action);
                moves += 1;
                if outcome.terminated {
                    break;
                }
            }

            let max_t = state.max_tile();
            let score = state.score;
            stats.scores.push(score);
            stats.max_tiles.push(max_t);
            stats.moves_counts.push(moves);

            if verbose {
                println!(
                    "  [Board {:>3}/{}] Score: {:>6} | Max Tile: {:>4} | Moves: {:>4}",
                    b + 1,
                    num_boards,
                    score,
                    max_t,
                    moves
                );
            }
        }

        stats.total_duration_secs = agent_start.elapsed().as_secs_f64();
        println!(
            "    Completed in {:.2}s (Avg Score: {:.1}, Max Tile: {})\n",
            stats.total_duration_secs,
            stats.avg_score(),
            stats.max_tiles.iter().copied().max().unwrap_or(0)
        );
        all_stats.push(stats);
    }

    let total_elapsed = arena_start.elapsed().as_secs_f64();

    // Sort by Avg Score descending
    all_stats.sort_by(|a, b| b.avg_score().partial_cmp(&a.avg_score()).unwrap());

    println!(
        "======================================================================================================================="
    );
    println!(
        "                                               FINAL TOURNAMENT RESULTS                                                "
    );
    println!(
        "======================================================================================================================="
    );
    println!(
        "Total Duration: {:.2}s | Standardized Boards: {}",
        total_elapsed, num_boards
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<22} | {:>9} | {:>9} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>9} | {:>8}",
        "Agent",
        "Avg Score",
        "Max Score",
        ">=16k",
        ">=8192",
        ">=4096",
        ">=2048",
        ">=1024",
        ">=512",
        ">=256",
        "Avg Moves",
        "Moves/s"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------"
    );

    for st in &all_stats {
        println!(
            "{:<22} | {:>9.1} | {:>9} | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>9.1} | {:>8.1}",
            st.name,
            st.avg_score(),
            st.max_score(),
            st.pct_tile(16384),
            st.pct_tile(8192),
            st.pct_tile(4096),
            st.pct_tile(2048),
            st.pct_tile(1024),
            st.pct_tile(512),
            st.pct_tile(256),
            st.avg_moves(),
            st.moves_per_sec()
        );
    }
    println!(
        "======================================================================================================================="
    );
}
