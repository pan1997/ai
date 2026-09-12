//! Benchmarking tournament arena runner for Hex.
//!
//! Evaluates win rates, first-mover advantage, tactical efficacy, and throughput across arbitrary
//! lists of agents ($K \ge 2$). Runs a complete round-robin tournament where every pair of agents
//! plays the specified number of games with strictly balanced seat allocations (half Black, half White).
//!
//! # Usage Examples
//! ```bash
//! # Fast 7x7 round-robin tournament across 4 agents
//! cargo run --release -p hex --bin hex-tournament -- --size 7 --agents heuristic,mcts-h:200,mcts:200,random --games 10
//!
//! # Standard 11x11 benchmark
//! cargo run --release -p hex --bin hex-tournament -- --size 11 --agents heuristic,mcts-h:500,random --games 4
//! ```

use hex::agent::HexAgentSpec;
use hex::game::{HexPlayer, HexState};
use hex::world::HexWorld;
use mcts_engine::arena::{GameOutcome, H2HMatrix, TwoPlayerTournamentStats, disambiguate_names};
use std::env;
use std::time::Instant;

fn print_help() {
    println!(
        r#"Hex - Round-Robin Tournament Arena

USAGE:
    hex-tournament [OPTIONS]

OPTIONS:
    --agents <SPECS> (or --players)
                             Comma-separated list of agent specifications.
                             Example: --agents heuristic,mcts-h:500,mcts:500,random
    --agent <SPEC> (or --player)
                             Repeatable flag to add an individual agent.
                             Example: --agent heuristic --agent mcts-h:500 --agent random
    --games <N>              Games to play per pairwise matchup [default: 10]
                             (half played as Black, half played as White)
    --size <N>               Board dimension N (supports 7, 9, 11) [default: 11]
    --pie-rule               Enable the Pie (Swap) rule on Move 2 for White
    --verbose                Print per-game move progression
    -h, --help               Print help information

SUPPORTED AGENT SPECS:
    heuristic                  1-ply greedy lookahead using shortest-path connectivity
    random                     Uniform random player
    mcts-h[:iters]             MCTS guided by 0-1 BFS shortest-path heuristic
    mcts[:iters[:rollouts]]    MCTS guided by random rollout simulations
    mcts-u[:iters]             MCTS with uniform priors and zero value baseline
"#
    );
}

fn run_game<const N: usize>(
    spec_black: &HexAgentSpec,
    spec_white: &HexAgentSpec,
    name_black: &str,
    name_white: &str,
    pie_rule: bool,
    verbose: bool,
) -> (GameOutcome, usize, usize) {
    let mut agent_black = spec_black.instantiate::<N>(name_black.to_string(), false);
    let mut agent_white = spec_white.instantiate::<N>(name_white.to_string(), false);

    let world = HexWorld::<N>::with_pie_rule(pie_rule);
    let mut state = HexState::<N>::with_pie_rule(pie_rule);
    let mut moves_black = 0;
    let mut moves_white = 0;

    while !world.is_terminal(&state) {
        let action = match state.current_player {
            HexPlayer::Black => {
                moves_black += 1;
                agent_black.select_action(&state)
            }
            HexPlayer::White => {
                moves_white += 1;
                agent_white.select_action(&state)
            }
        };

        let outcome = world.step_action(&mut state, action);

        if outcome.terminated {
            let result = if outcome.reward[0] > 0.0 {
                GameOutcome::Seat0Wins
            } else if outcome.reward[1] > 0.0 {
                GameOutcome::Seat1Wins
            } else {
                GameOutcome::Draw
            };

            if verbose {
                let winner_name = match result {
                    GameOutcome::Seat0Wins => name_black,
                    GameOutcome::Seat1Wins => name_white,
                    GameOutcome::Draw => "Draw",
                };
                let total_moves = moves_black + moves_white;
                println!("    Game finished in {total_moves:>3} moves: Winner = {winner_name}");
            }

            return (result, moves_black, moves_white);
        }
    }

    (GameOutcome::Draw, moves_black, moves_white)
}

fn run_tournament<const N: usize>(
    specs: &[HexAgentSpec],
    games_per_pair: usize,
    pie_rule: bool,
    verbose: bool,
) {
    let raw_names: Vec<String> = specs.iter().map(|s| s.default_name()).collect();
    let names = disambiguate_names(&raw_names);
    let num_agents = specs.len();

    let mut matrix = H2HMatrix::new(num_agents);
    let mut stats = vec![TwoPlayerTournamentStats::default(); num_agents];

    let num_matchups = num_agents * (num_agents - 1) / 2;
    let total_games_scheduled = num_matchups * games_per_pair;

    println!(
        "=========================================================================================="
    );
    println!(
        "                                 HEX TOURNAMENT ARENA ({}x{})",
        N, N
    );
    println!(
        "=========================================================================================="
    );
    println!(
        "Agents: {} | Matchups: {} | Games/Matchup: {} | Total Games: {} | Pie Rule: {}",
        num_agents,
        num_matchups,
        games_per_pair,
        total_games_scheduled,
        if pie_rule { "Enabled" } else { "Disabled" }
    );
    println!(
        "------------------------------------------------------------------------------------------"
    );

    let tournament_start = Instant::now();
    let mut total_moves: usize = 0;
    let mut matchup_idx = 0;

    for i in 0..num_agents {
        for j in (i + 1)..num_agents {
            matchup_idx += 1;
            let p1_name = &names[i];
            let p2_name = &names[j];

            println!(
                "\n>>> [{}/{}] Matchup: {} vs {}",
                matchup_idx, num_matchups, p1_name, p2_name
            );

            let matchup_start = Instant::now();
            let mut p1_wins = 0;
            let mut p2_wins = 0;
            let mut draws = 0;

            for g in 0..games_per_pair {
                let (outcome, m_black, m_white) = if g % 2 == 0 {
                    // Agent i is Black (Seat 0), Agent j is White (Seat 1)
                    let (res, mb, mw) =
                        run_game::<N>(&specs[i], &specs[j], p1_name, p2_name, pie_rule, verbose);
                    matrix.record_game(i, j, res);
                    TwoPlayerTournamentStats::record_game(&mut stats, i, j, res, mb, mw);
                    match res {
                        GameOutcome::Seat0Wins => p1_wins += 1,
                        GameOutcome::Seat1Wins => p2_wins += 1,
                        GameOutcome::Draw => draws += 1,
                    }
                    (res, mb, mw)
                } else {
                    // Agent j is Black (Seat 0), Agent i is White (Seat 1)
                    let (res, mb, mw) =
                        run_game::<N>(&specs[j], &specs[i], p2_name, p1_name, pie_rule, verbose);
                    matrix.record_game(j, i, res);
                    TwoPlayerTournamentStats::record_game(&mut stats, j, i, res, mb, mw);
                    match res {
                        GameOutcome::Seat0Wins => p2_wins += 1,
                        GameOutcome::Seat1Wins => p1_wins += 1,
                        GameOutcome::Draw => draws += 1,
                    }
                    (res, mb, mw)
                };
                total_moves += m_black + m_white;
                let _ = outcome;
            }

            let elapsed = matchup_start.elapsed().as_secs_f32();
            println!(
                "    Result: {} {}-{}-{} {} ({:.2}s)",
                p1_name, p1_wins, p2_wins, draws, p2_name, elapsed
            );
        }
    }

    let total_duration = tournament_start.elapsed().as_secs_f32();

    matrix.print_table(&names);
    TwoPlayerTournamentStats::print_standings(&stats, &names, "Black", "White");

    println!("Total Matchups:   {}", num_matchups);
    println!("Total Games:      {}", total_games_scheduled);
    println!("Elapsed Time:     {:.2}s", total_duration);
    println!("Total Moves:      {}", total_moves);
    println!(
        "Throughput:       {:.1} moves/sec",
        total_moves as f32 / total_duration.max(0.001)
    );
    println!(
        "=========================================================================================="
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut specs = Vec::new();
    let mut games_per_pair = 10;
    let mut size: usize = 11;
    let mut pie_rule = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "--agents" | "--players" => {
                i += 1;
                if i < args.len() {
                    for item in args[i].split(',') {
                        let trimmed = item.trim();
                        if !trimmed.is_empty() {
                            match HexAgentSpec::parse(trimmed) {
                                Ok(s) => specs.push(s),
                                Err(e) => {
                                    eprintln!("Error: {e}");
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
                    match HexAgentSpec::parse(&args[i]) {
                        Ok(s) => specs.push(s),
                        Err(e) => {
                            eprintln!("Error: {e}");
                            return;
                        }
                    }
                }
            }
            "--games" => {
                i += 1;
                if i < args.len() {
                    games_per_pair = args[i].parse().unwrap_or(10);
                }
            }
            "--size" => {
                i += 1;
                if i < args.len() {
                    size = args[i].parse().unwrap_or(11);
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

    if specs.len() < 2 {
        println!("At least 2 agents are required to run a tournament. Using defaults.");
        specs = vec![
            HexAgentSpec::parse("heuristic").unwrap(),
            HexAgentSpec::parse("mcts-h:200").unwrap(),
            HexAgentSpec::parse("random").unwrap(),
        ];
    }

    match size {
        7 => run_tournament::<7>(&specs, games_per_pair, pie_rule, verbose),
        9 => run_tournament::<9>(&specs, games_per_pair, pie_rule, verbose),
        11 => run_tournament::<11>(&specs, games_per_pair, pie_rule, verbose),
        other => {
            eprintln!("Unsupported size: {other}. Supported: 7, 9, 11");
        }
    }
}
