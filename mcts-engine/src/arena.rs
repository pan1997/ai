//! Tournament arena, head-to-head metrics, and leaderboard utilities.
//!
//! Provides game-agnostic round-robin tournament execution, head-to-head matrix tracking,
//! seat-fairness accounting, and standardized ASCII standings rendering.

use mcts_traits::{Agent, TurnBasedWorld};
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

/// Detailed outcome of executing a match between agents in a turn-based world environment.
#[derive(Debug, Clone)]
pub struct MatchResult<State, Reward> {
    /// Final world state at game termination.
    pub final_state: State,
    /// Total moves played across the match.
    pub total_moves: usize,
    /// Number of moves made by each seat index.
    pub moves_per_seat: Vec<usize>,
    /// Total wall-clock duration of the match.
    pub duration: std::time::Duration,
    /// Immediate terminal reward vector returned by the world at game end.
    pub final_reward: Reward,
    /// Convenience two-player outcome (`Seat0Wins`, `Seat1Wins`, `Draw`) if applicable.
    pub outcome_2p: Option<GameOutcome>,
    /// Index of the winning seat (or `None` if tied / draw).
    pub winner_seat: Option<usize>,
}

/// Generic match driver for executing turn-based games between agents.
///
/// Handles:
/// - Initializing world state and invoking [`Agent::reset`] across all participants.
/// - Maintaining per-agent transition history queues for [`Agent::select_action_with_history`].
/// - Enforcing turn-order arbitration via [`TurnBasedWorld::current_player`].
/// - Timing matches and compiling comprehensive [`MatchResult`] telemetry.
#[derive(Debug, Clone, Default)]
pub struct MatchDriver {
    /// Optional upper bound on total moves before aborting as a draw.
    pub max_moves: Option<usize>,
}

impl MatchDriver {
    /// Constructs a new `MatchDriver` without move limits.
    pub const fn new() -> Self {
        Self { max_moves: None }
    }

    /// Configures an upper limit on total moves.
    pub const fn with_max_moves(mut self, max_moves: usize) -> Self {
        self.max_moves = Some(max_moves);
        self
    }

    /// Plays a 2-player turn-based match between two agents.
    ///
    /// Automatically derives the 2-player outcome (`Seat0Wins`, `Seat1Wins`, `Draw`)
    /// by comparing `reward[0]` and `reward[1]`.
    pub fn play_2p<W, Act>(
        &self,
        world: &W,
        agent_0: &mut (dyn Agent<W::WorldState, Act> + '_),
        agent_1: &mut (dyn Agent<W::WorldState, Act> + '_),
        initial_state: Option<W::WorldState>,
    ) -> MatchResult<W::WorldState, [f32; 2]>
    where
        W: TurnBasedWorld<Action = Act, StepReward = [f32; 2]>,
        Act: Clone,
    {
        self.play_2p_with_callback(world, agent_0, agent_1, initial_state, |_, _, _| {})
    }

    /// Plays a 2-player turn-based match with a per-step callback.
    pub fn play_2p_with_callback<W, Act, F>(
        &self,
        world: &W,
        agent_0: &mut (dyn Agent<W::WorldState, Act> + '_),
        agent_1: &mut (dyn Agent<W::WorldState, Act> + '_),
        initial_state: Option<W::WorldState>,
        on_step: F,
    ) -> MatchResult<W::WorldState, [f32; 2]>
    where
        W: TurnBasedWorld<Action = Act, StepReward = [f32; 2]>,
        Act: Clone,
        F: FnMut(usize, &Act, &W::WorldState),
    {
        let mut agents: [&mut dyn Agent<W::WorldState, Act>; 2] = [agent_0, agent_1];
        self.play_multi_with_callback::<W, Act, 2, F>(world, &mut agents, initial_state, on_step)
    }

    /// Plays a single-player turn-based environment (e.g. 2048) until termination.
    pub fn play_single<W, Act>(
        &self,
        world: &W,
        agent: &mut (dyn Agent<W::WorldState, Act> + '_),
        initial_state: Option<W::WorldState>,
    ) -> MatchResult<W::WorldState, [f32; 1]>
    where
        W: TurnBasedWorld<Action = Act, StepReward = [f32; 1]>,
        Act: Clone,
    {
        let mut agents: [&mut dyn Agent<W::WorldState, Act>; 1] = [agent];
        self.play_multi_with_callback::<W, Act, 1, _>(
            world,
            &mut agents,
            initial_state,
            |_, _, _| {},
        )
    }

    /// Plays an $N$-player turn-based match across $P$ participating agents.
    pub fn play_multi<W, Act, const P: usize>(
        &self,
        world: &W,
        agents: &mut [&mut dyn Agent<W::WorldState, Act>],
        initial_state: Option<W::WorldState>,
    ) -> MatchResult<W::WorldState, [f32; P]>
    where
        W: TurnBasedWorld<Action = Act, StepReward = [f32; P]>,
        Act: Clone,
    {
        self.play_multi_with_callback::<W, Act, P, _>(world, agents, initial_state, |_, _, _| {})
    }

    /// Plays an $N$-player turn-based match across $P$ participating agents with a per-step callback.
    pub fn play_multi_with_callback<W, Act, const P: usize, F>(
        &self,
        world: &W,
        agents: &mut [&mut dyn Agent<W::WorldState, Act>],
        initial_state: Option<W::WorldState>,
        mut on_step: F,
    ) -> MatchResult<W::WorldState, [f32; P]>
    where
        W: TurnBasedWorld<Action = Act, StepReward = [f32; P]>,
        Act: Clone,
        F: FnMut(usize, &Act, &W::WorldState),
    {
        assert_eq!(
            agents.len(),
            P,
            "MatchDriver::play_multi: expected {P} agents, but received {}",
            agents.len()
        );

        for agent in agents.iter_mut() {
            agent.reset();
        }

        let start = std::time::Instant::now();
        let mut state = initial_state.unwrap_or_else(|| world.initial());
        let mut moves_per_seat = vec![0; P];
        let mut history_buffers: Vec<Vec<(Act, ())>> = (0..P).map(|_| Vec::new()).collect();
        let mut final_reward = [0.0f32; P];

        while !world.terminal(&state) {
            let total_moves: usize = moves_per_seat.iter().sum();
            if self.max_moves.is_some_and(|max| total_moves >= max) {
                break;
            }

            let active_seat = world.current_player(&state);
            assert!(
                active_seat < P,
                "TurnBasedWorld::current_player returned out-of-bounds seat {active_seat} (expected < {P})"
            );

            let action = agents[active_seat]
                .select_action_with_history(&state, &history_buffers[active_seat]);
            history_buffers[active_seat].clear();
            moves_per_seat[active_seat] += 1;

            let outcome = world.step_action(&mut state, &action);
            final_reward = outcome.reward;

            on_step(active_seat, &action, &state);

            for buf in &mut history_buffers {
                buf.push((action.clone(), ()));
            }

            if outcome.terminated {
                break;
            }
        }

        let total_moves: usize = moves_per_seat.iter().sum();
        let duration = start.elapsed();

        let outcome_2p = if P == 2 {
            if final_reward[0] > final_reward[1] {
                Some(GameOutcome::Seat0Wins)
            } else if final_reward[1] > final_reward[0] {
                Some(GameOutcome::Seat1Wins)
            } else {
                Some(GameOutcome::Draw)
            }
        } else {
            None
        };

        // Determine winner seat by highest reward (if unique)
        let mut best_score = f32::NEG_INFINITY;
        let mut best_seat = None;
        let mut tie = false;

        for (seat, &score) in final_reward.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_seat = Some(seat);
                tie = false;
            } else if (score - best_score).abs() < f32::EPSILON {
                tie = true;
            }
        }

        let winner_seat = if tie { None } else { best_seat };

        MatchResult {
            final_state: state,
            total_moves,
            moves_per_seat,
            duration,
            final_reward,
            outcome_2p,
            winner_seat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts_traits::World;

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

    struct Mock2PWorld;

    impl mcts_traits::World for Mock2PWorld {
        type WorldState = (usize, usize); // (current_player, total)
        type Action = usize;
        type Observation = (usize, usize);

        fn n_players(&self) -> usize {
            2
        }

        fn initial(&self) -> Self::WorldState {
            (0, 0)
        }

        fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
            *ws
        }

        fn actions(&self, _ws: &Self::WorldState, _player: usize, out: &mut Vec<Self::Action>) {
            out.clear();
            out.push(1);
            out.push(2);
        }

        fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
            let act = joint[ws.0];
            let outcome = self.step_action(ws, &act);
            (outcome.reward.to_vec(), outcome.terminated)
        }

        fn terminal(&self, ws: &Self::WorldState) -> bool {
            ws.1 >= 6
        }
    }

    impl TurnBasedWorld for Mock2PWorld {
        type StepReward = [f32; 2];

        fn current_player(&self, ws: &Self::WorldState) -> usize {
            ws.0
        }

        fn step_action(
            &self,
            ws: &mut Self::WorldState,
            action: &Self::Action,
        ) -> mcts_traits::StepOutcome<Self::StepReward> {
            ws.1 += action;
            let active = ws.0;
            let terminated = self.terminal(ws);
            if terminated {
                let reward = if active == 0 {
                    [1.0, -1.0]
                } else {
                    [-1.0, 1.0]
                };
                mcts_traits::StepOutcome::new(reward, true)
            } else {
                ws.0 = 1 - ws.0;
                mcts_traits::StepOutcome::new([0.0, 0.0], false)
            }
        }
    }

    struct HistoryTrackingAgent<S = (usize, usize)> {
        name: String,
        action_to_play: usize,
        history_seen: Vec<(usize, ())>,
        resets: usize,
        _marker: std::marker::PhantomData<fn(&S)>,
    }

    impl<S> HistoryTrackingAgent<S> {
        fn new(name: &str, action: usize) -> Self {
            Self {
                name: name.to_string(),
                action_to_play: action,
                history_seen: Vec::new(),
                resets: 0,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<S> Agent<S, usize> for HistoryTrackingAgent<S> {
        fn name(&self) -> &str {
            &self.name
        }

        fn select_action(&mut self, _state: &S) -> usize {
            self.action_to_play
        }

        fn select_action_with_history(&mut self, state: &S, history: &[(usize, ())]) -> usize {
            self.history_seen.extend_from_slice(history);
            self.select_action(state)
        }

        fn reset(&mut self) {
            self.resets += 1;
            self.history_seen.clear();
        }
    }

    #[test]
    fn test_match_driver_2p() {
        let world = Mock2PWorld;
        let mut p0 = HistoryTrackingAgent::new("P0", 2);
        let mut p1 = HistoryTrackingAgent::new("P1", 1);

        let driver = MatchDriver::new();
        let result = driver.play_2p(&world, &mut p0, &mut p1, None);

        // Moves sequence:
        // Turn 1 (P0): +2 -> total 2, p0 history []
        // Turn 2 (P1): +1 -> total 3, p1 history [(2, ())]
        // Turn 3 (P0): +2 -> total 5, p0 history [(2, ()), (1, ())]
        // Turn 4 (P1): +1 -> total 6 -> terminal! Winner is P1
        assert_eq!(result.total_moves, 4);
        assert_eq!(result.moves_per_seat, vec![2, 2]);
        assert_eq!(result.final_state, (1, 6));
        assert_eq!(result.final_reward, [-1.0, 1.0]);
        assert_eq!(result.outcome_2p, Some(GameOutcome::Seat1Wins));
        assert_eq!(result.winner_seat, Some(1));
        assert_eq!(p0.resets, 1);
        assert_eq!(p1.resets, 1);
        assert!(!p0.history_seen.is_empty());
        assert!(!p1.history_seen.is_empty());
    }

    #[test]
    fn test_match_driver_max_moves() {
        let world = Mock2PWorld;
        let mut p0 = HistoryTrackingAgent::new("P0", 1);
        let mut p1 = HistoryTrackingAgent::new("P1", 1);

        // Terminate early after 2 moves
        let driver = MatchDriver::new().with_max_moves(2);
        let result = driver.play_2p(&world, &mut p0, &mut p1, None);

        assert_eq!(result.total_moves, 2);
        assert_eq!(result.outcome_2p, Some(GameOutcome::Draw));
    }

    struct Mock1PWorld;

    impl mcts_traits::World for Mock1PWorld {
        type WorldState = usize;
        type Action = usize;
        type Observation = usize;

        fn n_players(&self) -> usize {
            1
        }

        fn initial(&self) -> Self::WorldState {
            0
        }

        fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
            *ws
        }

        fn actions(&self, _ws: &Self::WorldState, _player: usize, out: &mut Vec<Self::Action>) {
            out.clear();
            out.push(1);
        }

        fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
            *ws += joint.first().copied().unwrap_or(0);
            let term = *ws >= 5;
            (vec![if term { 10.0 } else { 1.0 }], term)
        }

        fn terminal(&self, ws: &Self::WorldState) -> bool {
            *ws >= 5
        }
    }

    impl TurnBasedWorld for Mock1PWorld {
        type StepReward = [f32; 1];

        fn current_player(&self, _ws: &Self::WorldState) -> usize {
            0
        }

        fn step_action(
            &self,
            ws: &mut Self::WorldState,
            action: &Self::Action,
        ) -> mcts_traits::StepOutcome<Self::StepReward> {
            *ws += action;
            let term = *ws >= 5;
            mcts_traits::StepOutcome::new([if term { 10.0 } else { 1.0 }], term)
        }
    }

    #[test]
    fn test_match_driver_single() {
        let world = Mock1PWorld;
        let mut agent = HistoryTrackingAgent::<usize>::new("Single", 1);

        let driver = MatchDriver::new();
        let result = driver.play_single(&world, &mut agent, None);

        assert_eq!(result.total_moves, 5);
        assert_eq!(result.final_state, 5);
        assert_eq!(result.final_reward, [10.0]);
        assert_eq!(result.winner_seat, Some(0));
        assert_eq!(agent.resets, 1);
    }

    #[test]
    fn test_match_driver_multi() {
        let world = Mock2PWorld;
        let mut p0 = HistoryTrackingAgent::new("P0", 2);
        let mut p1 = HistoryTrackingAgent::new("P1", 2);

        let driver = MatchDriver::new();
        let mut agents: [&mut dyn Agent<(usize, usize), usize>; 2] = [&mut p0, &mut p1];
        let result = driver.play_multi::<Mock2PWorld, usize, 2>(&world, &mut agents, None);

        // Turn 1 (P0): +2 -> total 2
        // Turn 2 (P1): +2 -> total 4
        // Turn 3 (P0): +2 -> total 6 -> terminal! P0 wins
        assert_eq!(result.total_moves, 3);
        assert_eq!(result.moves_per_seat, vec![2, 1]);
        assert_eq!(result.winner_seat, Some(0));
        assert_eq!(p0.resets, 1);
        assert_eq!(p1.resets, 1);
    }
}
