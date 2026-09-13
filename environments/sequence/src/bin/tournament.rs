//! Benchmarking tournament arena runner for Sequence.
//!
//! Evaluates win rates, game lengths, and head-to-head performance across diverse agents
//! (ISMCTS, Opponent-Model MCTS, Heuristic, Random) with balanced seat rotations.

use mcts_engine::arena::{
    disambiguate_names, GameOutcome, H2HMatrix, MatchDriver, TwoPlayerTournamentStats,
};
use sequence::agent::{parse_agent, BoxAgent};
use sequence::game::SequenceConfig;
use sequence::world::SequenceWorld;
use std::env;
use std::time::Instant;

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
    --players <N>       Number of players per game: 2 [default]
    --p1, --p2          Pairwise shorthand specs (e.g. --p1 is-mcts:300:5 --p2 heuristic)
    -h, --help          Print help information

AGENT SPECIFICATIONS:
    is-mcts[:iters[:dets]]             Information Set MCTS (e.g. is-mcts:300:5)
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
        "Agents ({}): {} | Games per pair: {}\n",
        n,
        names.join(", "),
        games_per_pair
    );

    let world = SequenceWorld::<2>::new(SequenceConfig::new_2p());
    let driver = MatchDriver::new();

    let mut h2h = H2HMatrix::new(n);
    let mut stats = vec![TwoPlayerTournamentStats::default(); n];
    let total_start = Instant::now();

    for i in 0..n {
        for j in (i + 1)..n {
            let half = games_per_pair / 2;
            let rem = games_per_pair - half;

            print!(
                "Running matchup: {} vs {} ({} games)... ",
                names[i], names[j], games_per_pair
            );

            // Games with agent i as Seat 0 (Player 0) and agent j as Seat 1 (Player 1)
            for _ in 0..half {
                let mut a0: BoxAgent<2> = parse_agent::<2>(&agent_specs[i], &names[i]);
                let mut a1: BoxAgent<2> = parse_agent::<2>(&agent_specs[j], &names[j]);

                let result = driver.play_2p(&world, a0.as_mut(), a1.as_mut(), None);
                let outcome = if result.final_reward[0] > 0.0 {
                    GameOutcome::Seat0Wins
                } else if result.final_reward[1] > 0.0 {
                    GameOutcome::Seat1Wins
                } else {
                    GameOutcome::Draw
                };

                h2h.record_game(i, j, outcome);
                TwoPlayerTournamentStats::record_game(
                    &mut stats,
                    i,
                    j,
                    outcome,
                    result.moves_per_seat[0],
                    result.moves_per_seat[1],
                );
            }

            // Games with agent j as Seat 0 and agent i as Seat 1 (swapped seats)
            for _ in 0..rem {
                let mut a0: BoxAgent<2> = parse_agent::<2>(&agent_specs[j], &names[j]);
                let mut a1: BoxAgent<2> = parse_agent::<2>(&agent_specs[i], &names[i]);

                let result = driver.play_2p(&world, a0.as_mut(), a1.as_mut(), None);
                let outcome = if result.final_reward[0] > 0.0 {
                    GameOutcome::Seat0Wins
                } else if result.final_reward[1] > 0.0 {
                    GameOutcome::Seat1Wins
                } else {
                    GameOutcome::Draw
                };

                // Note: swapped order for recording in matrix
                h2h.record_game(j, i, outcome);
                TwoPlayerTournamentStats::record_game(
                    &mut stats,
                    j,
                    i,
                    outcome,
                    result.moves_per_seat[0],
                    result.moves_per_seat[1],
                );
            }

            println!("done.");
        }
    }

    let elapsed = total_start.elapsed();
    println!("\nTournament completed in {:.2?}!\n", elapsed);

    // Render standings and H2H table
    TwoPlayerTournamentStats::print_standings(&stats, &names, "Seat 0 (Player 0)", "Seat 1 (Player 1)");
    h2h.print_table(&names);
}
