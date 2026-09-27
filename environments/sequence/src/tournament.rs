//! Parallel round-robin tournament execution for Sequence agents.
//!
//! Provides utilities for running matches between Sequence agents sequentially or
//! concurrently across multiple CPU worker threads using `std::thread::scope`.

use crate::agent::{BoxAgent, parse_agent};
use crate::game::SequenceConfig;
use crate::world::SequenceWorld;
use mcts_engine::arena::{GameOutcome, H2HMatrix, MatchDriver, TwoPlayerTournamentStats};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Seat assignment for a two-player matchup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatAssignment {
    /// Agent A is Seat 0 (Player 0 / First mover), Agent B is Seat 1.
    Seat0IsA,
    /// Agent B is Seat 0 (Player 0 / First mover), Agent A is Seat 1.
    Seat0IsB,
}

/// Telemetry result of a single two-player Sequence game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TournamentGameResult {
    /// Outcome from the perspective of Seat 0 vs Seat 1.
    pub outcome: GameOutcome,
    /// Number of moves made by Seat 0 (Player 0).
    pub moves_seat0: usize,
    /// Number of moves made by Seat 1 (Player 1).
    pub moves_seat1: usize,
}

/// Executes a single two-player Sequence game between two agent specifications.
///
/// # Arguments
/// - `spec_seat0`: Specification string for Seat 0 agent (e.g. `"is-mcts:200:4"`, `"heuristic"`).
/// - `name_seat0`: Display name for Seat 0 agent.
/// - `spec_seat1`: Specification string for Seat 1 agent.
/// - `name_seat1`: Display name for Seat 1 agent.
///
/// # Returns
/// A [`TournamentGameResult`] detailing the game outcome and move counts per seat.
#[must_use]
pub fn play_single_game(
    spec_seat0: &str,
    name_seat0: &str,
    spec_seat1: &str,
    name_seat1: &str,
) -> TournamentGameResult {
    let mut a0: BoxAgent<2> = parse_agent::<2>(spec_seat0, name_seat0);
    let mut a1: BoxAgent<2> = parse_agent::<2>(spec_seat1, name_seat1);
    let world = SequenceWorld::<2>::new(SequenceConfig::new_2p());
    let driver = MatchDriver::new();
    let result = driver.play_2p(&world, a0.as_mut(), a1.as_mut(), None);

    let outcome = if result.final_reward[0] > 0.0 {
        GameOutcome::Seat0Wins
    } else if result.final_reward[1] > 0.0 {
        GameOutcome::Seat1Wins
    } else {
        GameOutcome::Draw
    };

    TournamentGameResult {
        outcome,
        moves_seat0: result.moves_per_seat[0],
        moves_seat1: result.moves_per_seat[1],
    }
}

/// Executes a balanced two-player matchup between two agents, running up to `n_cpu` games in parallel.
///
/// Splits `total_games` into half with Agent A as Seat 0, and the remaining half with Agent B as Seat 0.
/// When `n_cpu > 1`, games are distributed dynamically across worker threads using `std::thread::scope`.
///
/// # Arguments
/// - `spec_a`: Agent A specification string.
/// - `name_a`: Agent A display name.
/// - `spec_b`: Agent B specification string.
/// - `name_b`: Agent B display name.
/// - `total_games`: Total number of games to play in this matchup.
/// - `n_cpu`: Number of worker threads to execute games in parallel. Clamped to `1..=total_games`.
///
/// # Returns
/// A vector of `(SeatAssignment, TournamentGameResult)` pairs for all played games.
pub fn play_matchup_parallel(
    spec_a: &str,
    name_a: &str,
    spec_b: &str,
    name_b: &str,
    total_games: usize,
    n_cpu: usize,
) -> Vec<(SeatAssignment, TournamentGameResult)> {
    if total_games == 0 {
        return Vec::new();
    }

    let half = total_games / 2;
    let rem = total_games - half;

    // Interleave assignments for balanced thread scheduling
    let mut tasks = Vec::with_capacity(total_games);
    let max_rounds = half.max(rem);
    for r in 0..max_rounds {
        if r < half {
            tasks.push(SeatAssignment::Seat0IsA);
        }
        if r < rem {
            tasks.push(SeatAssignment::Seat0IsB);
        }
    }

    let n_workers = n_cpu.clamp(1, total_games);

    if n_workers == 1 {
        return tasks
            .into_iter()
            .map(|assignment| {
                let res = match assignment {
                    SeatAssignment::Seat0IsA => play_single_game(spec_a, name_a, spec_b, name_b),
                    SeatAssignment::Seat0IsB => play_single_game(spec_b, name_b, spec_a, name_a),
                };
                (assignment, res)
            })
            .collect();
    }

    let tasks_mutex = Arc::new(Mutex::new(tasks));
    let (tx, rx) = mpsc::channel();

    thread::scope(|s| {
        for _ in 0..n_workers {
            let q = Arc::clone(&tasks_mutex);
            let t = tx.clone();
            s.spawn(move || {
                loop {
                    let task = {
                        let mut lock = q.lock().expect("task queue mutex poisoned");
                        lock.pop()
                    };
                    match task {
                        Some(assignment) => {
                            let res = match assignment {
                                SeatAssignment::Seat0IsA => {
                                    play_single_game(spec_a, name_a, spec_b, name_b)
                                }
                                SeatAssignment::Seat0IsB => {
                                    play_single_game(spec_b, name_b, spec_a, name_a)
                                }
                            };
                            t.send((assignment, res))
                                .expect("failed sending game result across thread channel");
                        }
                        None => break,
                    }
                }
            });
        }
        drop(tx);

        let mut results = Vec::with_capacity(total_games);
        for res in rx {
            results.push(res);
        }
        results
    })
}

/// Executes a complete round-robin tournament across multiple agents with parallel execution.
///
/// Runs all pairwise matchups with balanced seat rotations. Results are aggregated into
/// [`H2HMatrix`] and [`TwoPlayerTournamentStats`].
///
/// # Arguments
/// - `agent_specs`: List of agent specification strings.
/// - `names`: Disambiguated display names corresponding to `agent_specs`.
/// - `games_per_pair`: Number of games to play per pairwise matchup.
/// - `n_cpu`: Number of games to run in parallel per matchup.
/// - `on_matchup_start`: Callback invoked when a matchup begins `(i, j, name_a, name_b)`.
/// - `on_matchup_done`: Callback invoked when a matchup completes `(i, j, name_a, name_b, duration)`.
///
/// # Returns
/// A tuple containing `(H2HMatrix, Vec<TwoPlayerTournamentStats>, Duration)`.
pub fn run_tournament(
    agent_specs: &[String],
    names: &[String],
    games_per_pair: usize,
    n_cpu: usize,
    mut on_matchup_start: impl FnMut(usize, usize, &str, &str),
    mut on_matchup_done: impl FnMut(usize, usize, &str, &str, Duration),
) -> (H2HMatrix, Vec<TwoPlayerTournamentStats>, Duration) {
    let n = agent_specs.len();
    let mut h2h = H2HMatrix::new(n);
    let mut stats = vec![TwoPlayerTournamentStats::default(); n];
    let tournament_start = Instant::now();

    for i in 0..n {
        for j in (i + 1)..n {
            on_matchup_start(i, j, &names[i], &names[j]);
            let matchup_start = Instant::now();

            let results = play_matchup_parallel(
                &agent_specs[i],
                &names[i],
                &agent_specs[j],
                &names[j],
                games_per_pair,
                n_cpu,
            );

            for (assignment, res) in results {
                match assignment {
                    SeatAssignment::Seat0IsA => {
                        h2h.record_game(i, j, res.outcome);
                        TwoPlayerTournamentStats::record_game(
                            &mut stats,
                            i,
                            j,
                            res.outcome,
                            res.moves_seat0,
                            res.moves_seat1,
                        );
                    }
                    SeatAssignment::Seat0IsB => {
                        h2h.record_game(j, i, res.outcome);
                        TwoPlayerTournamentStats::record_game(
                            &mut stats,
                            j,
                            i,
                            res.outcome,
                            res.moves_seat0,
                            res.moves_seat1,
                        );
                    }
                }
            }

            on_matchup_done(i, j, &names[i], &names[j], matchup_start.elapsed());
        }
    }

    (h2h, stats, tournament_start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_play_single_game() {
        let res = play_single_game("random", "Random-0", "random", "Random-1");
        assert!(res.moves_seat0 + res.moves_seat1 > 0);
        assert!(matches!(
            res.outcome,
            GameOutcome::Seat0Wins | GameOutcome::Seat1Wins | GameOutcome::Draw
        ));
    }

    #[test]
    fn test_play_matchup_parallel() {
        let results = play_matchup_parallel("random", "R0", "random", "R1", 4, 2);
        assert_eq!(results.len(), 4);
        let seat_a_count = results
            .iter()
            .filter(|(a, _)| *a == SeatAssignment::Seat0IsA)
            .count();
        let seat_b_count = results
            .iter()
            .filter(|(a, _)| *a == SeatAssignment::Seat0IsB)
            .count();
        assert_eq!(seat_a_count, 2);
        assert_eq!(seat_b_count, 2);
    }

    #[test]
    fn test_run_tournament_parallel() {
        let specs = vec!["random".to_string(), "random".to_string()];
        let names = vec!["R0".to_string(), "R1".to_string()];
        let (h2h, stats, dur) = run_tournament(
            &specs,
            &names,
            2,
            2,
            |_i, _j, _a, _b| {},
            |_i, _j, _a, _b, _d| {},
        );
        assert_eq!(h2h.num_agents(), 2);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].games_played, 2);
        assert_eq!(stats[1].games_played, 2);
        assert!(dur > Duration::ZERO);
    }
}
