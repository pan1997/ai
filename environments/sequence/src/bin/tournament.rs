//! Benchmarking tournament arena runner for Sequence.
//!
//! Evaluates win rates, game lengths, and head-to-head performance across diverse agents
//! (ISMCTS, Opponent-Model MCTS, Heuristic, Random) with balanced seat rotations and
//! multi-core parallel game execution.

use mcts_engine::arena::{TwoPlayerTournamentStats, disambiguate_names};
use sequence::tournament::run_tournament;
use std::env;
use std::io::Write;

fn print_help() {
    println!(
        r#"Sequence - Round-Robin Tournament Arena

USAGE:
    sequence-tournament [OPTIONS]

OPTIONS:
    --agents <SPECS>    Comma-separated list of agent specifications
                        [default: is-mcts:200:4,macro-heuristic:200:4,heuristic,random]
    --agent <SPEC>      Repeatable flag to add an individual agent spec
    --games <N>         Number of games to play per pair [default: 10]
                        (half as Seat 0 / Team 0, half as Seat 1 / Team 1)
    -n-cpu, --n-cpu <N> Number of games to run in parallel [default: 1]
    --players <N>       Number of players per game: 2 [default]
    --p1, --p2          Pairwise shorthand specs (e.g. --p1 is-mcts:300:5 --p2 heuristic)
    -h, --help          Print help information

AGENT SPECIFICATIONS:
    is-mcts[:iters[:dets]]             Multi-Tree ISMCTS (e.g. is-mcts:300:5)
    is-mcts-single[:iters]             Single-Tree ISMCTS (e.g. is-mcts-single:300)
    macro-heuristic[:iters[:dets]]     Opponent-Model MCTS with heuristic opponent policy
    macro-random[:iters[:dets]]        Opponent-Model MCTS with random opponent policy
    mcts[:iters]                       Standard perfect-information MCTS
    heuristic                          1-ply greedy heuristic agent
    random                             Uniform random agent
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return;
    }

    let mut agent_specs: Vec<String> = Vec::new();
    let mut games_per_pair = 10usize;
    let mut n_cpu = 1usize;
    let mut p1_spec: Option<String> = None;
    let mut p2_spec: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agents" if i + 1 < args.len() => {
                for s in args[i + 1].split(',') {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        agent_specs.push(trimmed.to_string());
                    }
                }
                i += 2;
            }
            "--agent" if i + 1 < args.len() => {
                agent_specs.push(args[i + 1].clone());
                i += 2;
            }
            "--games" if i + 1 < args.len() => {
                games_per_pair = args[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            "-n-cpu" | "--n-cpu" | "-n" | "--cpus" | "--threads" if i + 1 < args.len() => {
                n_cpu = args[i + 1].parse().unwrap_or(1).max(1);
                i += 2;
            }
            arg if arg.starts_with("-n-cpu=") || arg.starts_with("--n-cpu=") => {
                if let Some((_, val)) = arg.split_once('=') {
                    n_cpu = val.parse().unwrap_or(1).max(1);
                }
                i += 1;
            }
            arg if arg.starts_with("--cpus=") || arg.starts_with("--threads=") => {
                if let Some((_, val)) = arg.split_once('=') {
                    n_cpu = val.parse().unwrap_or(1).max(1);
                }
                i += 1;
            }
            "--p1" if i + 1 < args.len() => {
                p1_spec = Some(args[i + 1].clone());
                i += 2;
            }
            "--p2" if i + 1 < args.len() => {
                p2_spec = Some(args[i + 1].clone());
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    if let (Some(p1), Some(p2)) = (p1_spec, p2_spec) {
        agent_specs = vec![p1, p2];
    }

    if agent_specs.is_empty() {
        agent_specs = vec![
            "is-mcts:200:4".to_string(),
            "macro-heuristic:200:4".to_string(),
            "heuristic".to_string(),
            "random".to_string(),
        ];
    }

    if agent_specs.len() < 2 {
        eprintln!("Tournament requires at least 2 agents.");
        return;
    }

    let names = disambiguate_names(&agent_specs);
    let n = agent_specs.len();

    println!("============================================================");
    println!("          🎴 SEQUENCE - ROUND-ROBIN TOURNAMENT 🎴          ");
    println!("============================================================");
    println!(
        "Agents ({}): {} | Games per pair: {} | Parallel workers (n-cpu): {}\n",
        n,
        names.join(", "),
        games_per_pair,
        n_cpu
    );

    let (h2h, stats, elapsed) = run_tournament(
        &agent_specs,
        &names,
        games_per_pair,
        n_cpu,
        |_i, _j, name_a, name_b| {
            print!(
                "Running matchup: {} vs {} ({} games)... ",
                name_a, name_b, games_per_pair
            );
            let _ = std::io::stdout().flush();
        },
        |_i, _j, _name_a, _name_b, _dur| {
            println!("done.");
        },
    );

    println!("\nTournament completed in {:.2?}!\n", elapsed);

    // Render standings and H2H table
    TwoPlayerTournamentStats::print_standings(
        &stats,
        &names,
        "Seat 0 (Player 0)",
        "Seat 1 (Player 1)",
    );
    h2h.print_table(&names);
}
