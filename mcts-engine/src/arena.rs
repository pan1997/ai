//! Tournament arena, head-to-head metrics, and leaderboard utilities.
//!
//! Provides game-agnostic round-robin tournament execution, head-to-head matrix tracking,
//! seat-fairness accounting, and standardized ASCII standings rendering.

use std::collections::HashMap;

/// Outcome of a two-player game from the perspective of Seat 0 vs Seat 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOutcome {
    /// Seat 0 (first player) won.
    Seat0Wins,
    /// Seat 1 (second player) won.
    Seat1Wins,
    /// Game ended in a draw / tie.
    Draw,
}

/// Disambiguates duplicate agent names by appending sequential numeric suffixes (e.g. `MCTS-1`, `MCTS-2`).
pub fn disambiguate_names<S: AsRef<str>>(raw_names: &[S]) -> Vec<String> {
    let mut counts = HashMap::new();
    for name in raw_names {
        *counts.entry(name.as_ref().to_string()).or_insert(0) += 1;
    }

    let mut current_idx = HashMap::new();
    let mut final_names = Vec::with_capacity(raw_names.len());

    for name in raw_names {
        let name_str = name.as_ref();
        if counts[name_str] > 1 {
            let idx = current_idx.entry(name_str.to_string()).or_insert(1);
            final_names.push(format!("{name_str}-{idx}"));
            *idx += 1;
        } else {
            final_names.push(name_str.to_string());
        }
    }

    final_names
}

/// Head-to-head match matrix tracking pairwise (wins, losses, draws) across $N$ agents.
#[derive(Debug, Clone)]
pub struct H2HMatrix {
    num_agents: usize,
    records: Vec<Vec<(usize, usize, usize)>>,
}

impl H2HMatrix {
    /// Creates a new empty $N \times N$ head-to-head matrix.
    pub fn new(num_agents: usize) -> Self {
        Self {
            num_agents,
            records: vec![vec![(0, 0, 0); num_agents]; num_agents],
        }
    }

    /// Number of agents participating in the matrix.
    #[inline]
    pub fn num_agents(&self) -> usize {
        self.num_agents
    }

    /// Records the outcome of a single match between `agent_0` (Seat 0) and `agent_1` (Seat 1).
    pub fn record_game(&mut self, agent_0: usize, agent_1: usize, outcome: GameOutcome) {
        match outcome {
            GameOutcome::Seat0Wins => {
                self.records[agent_0][agent_1].0 += 1;
                self.records[agent_1][agent_0].1 += 1;
            }
            GameOutcome::Seat1Wins => {
                self.records[agent_1][agent_0].0 += 1;
                self.records[agent_0][agent_1].1 += 1;
            }
            GameOutcome::Draw => {
                self.records[agent_0][agent_1].2 += 1;
                self.records[agent_1][agent_0].2 += 1;
            }
        }
    }

    /// Returns the pairwise `(wins, losses, draws)` of `agent_i` against `agent_j`.
    #[inline]
    pub fn get(&self, agent_i: usize, agent_j: usize) -> (usize, usize, usize) {
        self.records[agent_i][agent_j]
    }

    /// Returns the cumulative `(total_wins, total_losses, total_draws)` for `agent`.
    pub fn totals(&self, agent: usize) -> (usize, usize, usize) {
        let mut total_w = 0;
        let mut total_l = 0;
        let mut total_d = 0;
        for j in 0..self.num_agents {
            if j != agent {
                let (w, l, d) = self.records[agent][j];
                total_w += w;
                total_l += l;
                total_d += d;
            }
        }
        (total_w, total_l, total_d)
    }

    /// Returns the total number of games played by `agent`.
    #[inline]
    pub fn games_played(&self, agent: usize) -> usize {
        let (w, l, d) = self.totals(agent);
        w + l + d
    }

    /// Returns the overall win rate (percentage $[0.0, 100.0]$) for `agent`.
    pub fn win_rate(&self, agent: usize) -> f64 {
        let (w, _, _) = self.totals(agent);
        let total = self.games_played(agent);
        if total > 0 {
            (w as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Prints a formatted ASCII head-to-head cross table.
    pub fn print_table(&self, names: &[String]) {
        println!(
            "\n=========================================================================================="
        );
        println!(
            "                               HEAD-TO-HEAD MATRIX (W-L-D)                                "
        );
        println!(
            "=========================================================================================="
        );
        print!("{:<32} |", "Agent");
        for idx in 0..self.num_agents {
            print!(" [{:>2}]   |", idx + 1);
        }
        println!(" Total (W-L-D)   | Win Rate");
        println!("{}", "-".repeat(34 + self.num_agents * 9 + 28));

        for (i, name) in names.iter().enumerate().take(self.num_agents) {
            print!("[{:>2}] {:<28} |", i + 1, name);
            for j in 0..self.num_agents {
                if i == j {
                    print!("  ---   |");
                } else {
                    let (w, l, d) = self.records[i][j];
                    print!(" {:>2}-{:<2}-{:<1}|", w, l, d);
                }
            }
            let (total_w, total_l, total_d) = self.totals(i);
            let wr = self.win_rate(i);
            println!(
                " {:>3}-{:<3}-{:<2}  | {:>5.1}%",
                total_w, total_l, total_d, wr
            );
        }
    }
}

/// Cumulative match statistics for an individual agent in a 2-player tournament.
#[derive(Debug, Clone, Default)]
pub struct TwoPlayerTournamentStats {
    /// Total games played by this agent.
    pub games_played: usize,
    /// Total games won.
    pub wins: usize,
    /// Total games lost.
    pub losses: usize,
    /// Total games drawn.
    pub draws: usize,
    /// Games played in Seat 0 (e.g. Red / White / First player).
    pub seat0_games: usize,
    /// Wins achieved in Seat 0.
    pub seat0_wins: usize,
    /// Games played in Seat 1 (e.g. Yellow / Black / Second player).
    pub seat1_games: usize,
    /// Wins achieved in Seat 1.
    pub seat1_wins: usize,
    /// Total moves made across all games.
    pub total_moves: usize,
}

impl TwoPlayerTournamentStats {
    /// Records a completed game between `agent_0` and `agent_1`.
    pub fn record_game(
        stats: &mut [TwoPlayerTournamentStats],
        agent_0: usize,
        agent_1: usize,
        outcome: GameOutcome,
        moves_agent_0: usize,
        moves_agent_1: usize,
    ) {
        stats[agent_0].games_played += 1;
        stats[agent_0].seat0_games += 1;
        stats[agent_0].total_moves += moves_agent_0;

        stats[agent_1].games_played += 1;
        stats[agent_1].seat1_games += 1;
        stats[agent_1].total_moves += moves_agent_1;

        match outcome {
            GameOutcome::Seat0Wins => {
                stats[agent_0].wins += 1;
                stats[agent_0].seat0_wins += 1;
                stats[agent_1].losses += 1;
            }
            GameOutcome::Seat1Wins => {
                stats[agent_1].wins += 1;
                stats[agent_1].seat1_wins += 1;
                stats[agent_0].losses += 1;
            }
            GameOutcome::Draw => {
                stats[agent_0].draws += 1;
                stats[agent_1].draws += 1;
            }
        }
    }

    /// Prints a formatted ranked leaderboard sorted by win rate descending.
    pub fn print_standings(
        stats: &[TwoPlayerTournamentStats],
        names: &[String],
        seat0_label: &str,
        seat1_label: &str,
    ) {
        let mut rank_indices: Vec<usize> = (0..stats.len()).collect();
        rank_indices.sort_by(|&a, &b| {
            let wr_a = if stats[a].games_played > 0 {
                stats[a].wins as f64 / stats[a].games_played as f64
            } else {
                0.0
            };
            let wr_b = if stats[b].games_played > 0 {
                stats[b].wins as f64 / stats[b].games_played as f64
            } else {
                0.0
            };
            wr_b.partial_cmp(&wr_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        println!(
            "\n=========================================================================================="
        );
        println!(
            "                                    FINAL STANDINGS                                       "
        );
        println!(
            "=========================================================================================="
        );
        let seat0_col = format!("{seat0_label} W%");
        let seat1_col = format!("{seat1_label} W%");
        println!(
            "{:<4} {:<30} {:>6} {:>6} {:>6} {:>6} {:>8} {:>10} {:>10}",
            "Pos", "Agent", "Games", "Wins", "Loss", "Draw", "Win %", seat0_col, seat1_col
        );
        println!("{}", "-".repeat(92));

        for (pos, &idx) in rank_indices.iter().enumerate() {
            let s = &stats[idx];
            let win_pct = if s.games_played > 0 {
                (s.wins as f64 / s.games_played as f64) * 100.0
            } else {
                0.0
            };
            let seat0_pct = if s.seat0_games > 0 {
                (s.seat0_wins as f64 / s.seat0_games as f64) * 100.0
            } else {
                0.0
            };
            let seat1_pct = if s.seat1_games > 0 {
                (s.seat1_wins as f64 / s.seat1_games as f64) * 100.0
            } else {
                0.0
            };

            println!(
                "{:<4} {:<30} {:>6} {:>6} {:>6} {:>6} {:>7.1}% {:>9.1}% {:>9.1}%",
                pos + 1,
                names[idx],
                s.games_played,
                s.wins,
                s.losses,
                s.draws,
                win_pct,
                seat0_pct,
                seat1_pct
            );
        }
        println!(
            "==========================================================================================\n"
        );
    }
}

/// Cumulative match statistics for an individual agent in a multi-player ($P \ge 2$) tournament.
#[derive(Debug, Clone)]
pub struct MultiPlayerTournamentStats {
    /// Total games where this agent placed first (solo or tied).
    pub total_wins: usize,
    /// Games where this agent was the sole outright winner.
    pub solo_wins: usize,
    /// Games where this agent tied for first place.
    pub tied_wins: usize,
    /// Cumulative raw game score.
    pub total_score: i64,
    /// Secondary custom metric accumulator (e.g. pieces or squares placed).
    pub custom_metric_total: f64,
    /// Count of games assigned to each seat index $0 \le s < P$.
    pub seat_counts: Vec<usize>,
}

impl MultiPlayerTournamentStats {
    /// Creates a new `MultiPlayerTournamentStats` record configured for `num_seats`.
    pub fn new(num_seats: usize) -> Self {
        Self {
            total_wins: 0,
            solo_wins: 0,
            tied_wins: 0,
            total_score: 0,
            custom_metric_total: 0.0,
            seat_counts: vec![0; num_seats],
        }
    }

    /// Records the outcome of a multi-player game.
    ///
    /// - `seat_to_agent`: Mapping from seat index to agent index.
    /// - `scores`: Final score for each seat.
    /// - `custom_metrics`: Optional secondary metric for each seat.
    pub fn record_game(
        stats: &mut [MultiPlayerTournamentStats],
        seat_to_agent: &[usize],
        scores: &[i32],
        custom_metrics: Option<&[f64]>,
    ) {
        let num_seats = seat_to_agent.len();
        for (seat, &agent_idx) in seat_to_agent.iter().enumerate() {
            stats[agent_idx].seat_counts[seat] += 1;
            stats[agent_idx].total_score += scores[seat] as i64;
            if let Some(metrics) = custom_metrics {
                stats[agent_idx].custom_metric_total += metrics[seat];
            }
        }

        let max_score = scores.iter().copied().max().unwrap_or(0);
        let winning_seats: Vec<usize> =
            (0..num_seats).filter(|&s| scores[s] == max_score).collect();

        if winning_seats.len() == 1 {
            let agent_idx = seat_to_agent[winning_seats[0]];
            stats[agent_idx].total_wins += 1;
            stats[agent_idx].solo_wins += 1;
        } else {
            for &seat in &winning_seats {
                let agent_idx = seat_to_agent[seat];
                stats[agent_idx].total_wins += 1;
                stats[agent_idx].tied_wins += 1;
            }
        }
    }

    /// Prints a formatted ranked leaderboard sorted by total wins descending, then by total score descending.
    pub fn print_leaderboard(
        stats: &[MultiPlayerTournamentStats],
        names: &[String],
        total_games: usize,
        elapsed: std::time::Duration,
        custom_metric_label: Option<&str>,
    ) -> Vec<usize> {
        let n = total_games.max(1) as f64;
        let mut rank_indices: Vec<usize> = (0..stats.len()).collect();
        rank_indices.sort_by(|&a, &b| {
            stats[b]
                .total_wins
                .cmp(&stats[a].total_wins)
                .then_with(|| stats[b].total_score.cmp(&stats[a].total_score))
        });

        println!(
            "\n======================================== TOURNAMENT RESULTS ========================================"
        );
        println!(
            "Total Duration: {:.2?} | Games: {total_games} | Seating: Uniformly Shuffled Per Match",
            elapsed
        );
        println!(
            "----------------------------------------------------------------------------------------------------"
        );

        let custom_header = custom_metric_label.unwrap_or("Custom Metric");
        println!(
            "Agent                        | Total Wins | Solo Wins | Tied Wins | Win Rate | Avg Score | {:<12}",
            custom_header
        );
        println!(
            "----------------------------------------------------------------------------------------------------"
        );

        for &idx in &rank_indices {
            let st = &stats[idx];
            println!(
                "{:<28} | {:>10} | {:>9} | {:>9} | {:>7.1}% | {:>9.2} | {:>12.1}",
                names[idx],
                st.total_wins,
                st.solo_wins,
                st.tied_wins,
                (st.total_wins as f64 / n) * 100.0,
                (st.total_score as f64) / n,
                st.custom_metric_total / n,
            );
        }

        rank_indices
    }

    /// Prints a seat fairness audit table showing game counts per seat for each agent.
    pub fn print_seating_fairness(
        stats: &[MultiPlayerTournamentStats],
        names: &[String],
        rank_indices: &[usize],
        seat_names: &[&str],
    ) {
        println!(
            "----------------------------------------------------------------------------------------------------"
        );
        println!("Seating Distribution (Seat Fairness Check):");
        for &idx in rank_indices {
            let st = &stats[idx];
            let seat_strs: Vec<String> = st
                .seat_counts
                .iter()
                .enumerate()
                .map(|(s, &count)| {
                    let label = seat_names.get(s).copied().unwrap_or("?");
                    format!("Seat {s} ({label}): {count:2}")
                })
                .collect();
            println!("  {:<28} -> {}", names[idx], seat_strs.join(" | "));
        }
        println!(
            "====================================================================================================\n"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disambiguate_names() {
        let raw = vec!["mcts", "random", "mcts", "tactical", "mcts"];
        let unique = disambiguate_names(&raw);
        assert_eq!(
            unique,
            vec!["mcts-1", "random", "mcts-2", "tactical", "mcts-3"]
        );
    }

    #[test]
    fn test_h2h_matrix_and_stats() {
        let mut h2h = H2HMatrix::new(3);
        assert_eq!(h2h.num_agents(), 3);

        // Agent 0 vs Agent 1: Agent 0 wins
        h2h.record_game(0, 1, GameOutcome::Seat0Wins);
        // Agent 0 vs Agent 1: Agent 1 wins
        h2h.record_game(0, 1, GameOutcome::Seat1Wins);
        // Agent 0 vs Agent 2: Draw
        h2h.record_game(0, 2, GameOutcome::Draw);

        assert_eq!(h2h.get(0, 1), (1, 1, 0));
        assert_eq!(h2h.get(1, 0), (1, 1, 0));
        assert_eq!(h2h.get(0, 2), (0, 0, 1));
        assert_eq!(h2h.get(2, 0), (0, 0, 1));

        assert_eq!(h2h.totals(0), (1, 1, 1));
        assert_eq!(h2h.games_played(0), 3);
        assert!((h2h.win_rate(0) - 33.333).abs() < 0.1);
    }

    #[test]
    fn test_two_player_tournament_stats() {
        let mut stats = vec![TwoPlayerTournamentStats::default(); 2];
        TwoPlayerTournamentStats::record_game(&mut stats, 0, 1, GameOutcome::Seat0Wins, 10, 9);

        assert_eq!(stats[0].wins, 1);
        assert_eq!(stats[0].seat0_wins, 1);
        assert_eq!(stats[0].total_moves, 10);
        assert_eq!(stats[1].losses, 1);
        assert_eq!(stats[1].seat1_games, 1);
        assert_eq!(stats[1].total_moves, 9);
    }

    #[test]
    fn test_multi_player_tournament_stats() {
        let mut stats = vec![MultiPlayerTournamentStats::new(4); 4];
        let seat_to_agent = [0, 1, 2, 3];
        let scores = [20, 15, 20, 10]; // 0 and 2 tie for first place
        let metrics = [89.0, 70.0, 89.0, 50.0];

        MultiPlayerTournamentStats::record_game(
            &mut stats,
            &seat_to_agent,
            &scores,
            Some(&metrics),
        );

        assert_eq!(stats[0].total_wins, 1);
        assert_eq!(stats[0].tied_wins, 1);
        assert_eq!(stats[0].solo_wins, 0);
        assert_eq!(stats[0].total_score, 20);
        assert_eq!(stats[0].custom_metric_total, 89.0);
        assert_eq!(stats[0].seat_counts[0], 1);

        assert_eq!(stats[1].total_wins, 0);
        assert_eq!(stats[1].seat_counts[1], 1);

        assert_eq!(stats[2].total_wins, 1);
        assert_eq!(stats[2].tied_wins, 1);
        assert_eq!(stats[2].solo_wins, 0);

        assert_eq!(stats[3].total_wins, 0);
        assert_eq!(stats[3].seat_counts[3], 1);
    }
}
