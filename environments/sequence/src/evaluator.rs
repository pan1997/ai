//! Evaluators and heuristic estimators for Sequence MCTS planning.

use crate::board::{coord_to_index, is_corner_index, is_one_eyed_jack, is_two_eyed_jack};
use crate::game::{SequenceAction, SequenceState};
use mcts_traits::{Evaluation, Model};

/// Evaluator assigning uniform priors and zero values.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl Model<SequenceState> for UniformEvaluator {
    fn evaluate(&self, s: &SequenceState) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_actions(&mut legal);
        let n = legal.len();
        let priors = if n > 0 {
            vec![1.0 / (n as f32); n]
        } else {
            Vec::new()
        };
        let p = s.config.num_players;
        Evaluation {
            priors,
            values: vec![0.0; p],
        }
    }
}

/// Domain-specific heuristic evaluator for Sequence.
///
/// Evaluates positions by counting potential lines, sequence progress, corner proximity,
/// and Jack card advantages. Also provides direct 1-ply action scoring for greedy heuristic agents.
#[derive(Debug, Clone, Copy)]
pub struct SequenceHeuristicEvaluator {
    /// Weight for each completed sequence.
    pub weight_sequence: f32,
    /// Weight for an open 4-in-a-row threat.
    pub weight_threat_4: f32,
    /// Weight for an open 3-in-a-row development.
    pub weight_threat_3: f32,
    /// Weight for an open 2-in-a-row development.
    pub weight_threat_2: f32,
    /// Value assigned to holding a Two-Eyed Jack (wildcard).
    pub weight_two_eyed_jack: f32,
    /// Value assigned to holding a One-Eyed Jack (removal).
    pub weight_one_eyed_jack: f32,
}

impl Default for SequenceHeuristicEvaluator {
    fn default() -> Self {
        Self {
            weight_sequence: 50.0,
            weight_threat_4: 15.0,
            weight_threat_3: 4.0,
            weight_threat_2: 1.0,
            weight_two_eyed_jack: 3.5,
            weight_one_eyed_jack: 2.5,
        }
    }
}

impl SequenceHeuristicEvaluator {
    /// Creates a new heuristic evaluator with default weights.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            weight_sequence: 50.0,
            weight_threat_4: 15.0,
            weight_threat_3: 4.0,
            weight_threat_2: 1.0,
            weight_two_eyed_jack: 3.5,
            weight_one_eyed_jack: 2.5,
        }
    }

    /// Evaluates raw heuristic strength for team `team` on the board.
    #[must_use]
    pub fn team_score(&self, s: &SequenceState, team: u8) -> f32 {
        let mut score = s.team_sequence_counts[team as usize] as f32 * self.weight_sequence;
        if s.team_sequence_counts[team as usize] >= s.config.target_sequences {
            return score + 500.0;
        }

        // Helper to scan a 5-cell line
        let evaluate_line = |indices: [usize; 5]| -> f32 {
            let mut team_count = 0;
            let mut opp_count = 0;

            for &idx in &indices {
                if is_corner_index(idx) {
                    team_count += 1; // Free corner
                } else {
                    match s.board[idx] {
                        Some(t) if t == team => team_count += 1,
                        Some(_) => opp_count += 1,
                        None => {}
                    }
                }
            }

            if opp_count > 0 {
                0.0 // Blocked by opponent
            } else {
                match team_count {
                    4 => self.weight_threat_4,
                    3 => self.weight_threat_3,
                    2 => self.weight_threat_2,
                    _ => 0.0,
                }
            }
        };

        // Horizontal windows
        for r in 0..10 {
            for c in 0..=5 {
                let line = [
                    coord_to_index(r, c),
                    coord_to_index(r, c + 1),
                    coord_to_index(r, c + 2),
                    coord_to_index(r, c + 3),
                    coord_to_index(r, c + 4),
                ];
                score += evaluate_line(line);
            }
        }

        // Vertical windows
        for r in 0..=5 {
            for c in 0..10 {
                let line = [
                    coord_to_index(r, c),
                    coord_to_index(r + 1, c),
                    coord_to_index(r + 2, c),
                    coord_to_index(r + 3, c),
                    coord_to_index(r + 4, c),
                ];
                score += evaluate_line(line);
            }
        }

        // Diagonal windows (\)
        for r in 0..=5 {
            for c in 0..=5 {
                let line = [
                    coord_to_index(r, c),
                    coord_to_index(r + 1, c + 1),
                    coord_to_index(r + 2, c + 2),
                    coord_to_index(r + 3, c + 3),
                    coord_to_index(r + 4, c + 4),
                ];
                score += evaluate_line(line);
            }
        }

        // Anti-diagonal windows (/)
        for r in 0..=5 {
            for c in 4..10 {
                let line = [
                    coord_to_index(r, c),
                    coord_to_index(r + 1, c - 1),
                    coord_to_index(r + 2, c - 2),
                    coord_to_index(r + 3, c - 3),
                    coord_to_index(r + 4, c - 4),
                ];
                score += evaluate_line(line);
            }
        }

        // Card bonus for seated players of this team
        for (p, hand) in s.hands.iter().enumerate() {
            if s.config.player_team(p) == team {
                for &c in hand {
                    if is_two_eyed_jack(c) {
                        score += self.weight_two_eyed_jack;
                    } else if is_one_eyed_jack(c) {
                        score += self.weight_one_eyed_jack;
                    }
                }
            }
        }

        score
    }

    /// Evaluates a 1-ply candidate action from `state`, returning a heuristic quality score.
    #[must_use]
    pub fn score_action(&self, state: &SequenceState, action: &SequenceAction) -> f32 {
        let team = state.active_team();
        let current_sequences = state.team_sequence_counts[team as usize];

        // Simulate step on a light clone
        let mut sim = state.clone();
        let mut rng = rand::thread_rng();
        sim.step(action, &mut rng);

        // Immediate win / sequence completion bonus
        let new_sequences = sim.team_sequence_counts[team as usize];
        if new_sequences >= sim.config.target_sequences {
            return 10_000.0;
        }
        if new_sequences > current_sequences {
            return 1_000.0 * (new_sequences - current_sequences) as f32;
        }

        match action {
            SequenceAction::RemoveToken { pos, .. } => {
                // Removing an opponent piece that was part of an opponent threat
                let target_idx = coord_to_index(pos.0, pos.1);
                let opp_team = state.board[target_idx].unwrap_or(1 - team);
                let opp_score_before = self.team_score(state, opp_team);
                let opp_score_after = self.team_score(&sim, opp_team);
                let threat_reduced = opp_score_before - opp_score_after;
                50.0 + threat_reduced * 2.0
            }
            SequenceAction::PlayCard { pos, .. } => {
                let my_score_before = self.team_score(state, team);
                let my_score_after = self.team_score(&sim, team);
                let progress = my_score_after - my_score_before;

                // Extra bonus for corner proximity
                let corner_bonus = if is_corner_adjacent(pos.0, pos.1) {
                    5.0
                } else {
                    0.0
                };

                progress + corner_bonus
            }
            SequenceAction::DiscardDeadCard { .. } => 2.0,
        }
    }
}

/// Returns true if coordinate $(r, c)$ is directly adjacent to one of the four corner wild spaces.
#[inline]
fn is_corner_adjacent(r: u8, c: u8) -> bool {
    let corners = [(0, 0), (0, 9), (9, 0), (9, 9)];
    for &(cr, cc) in &corners {
        let dr = (r as i8 - cr as i8).abs();
        let dc = (c as i8 - cc as i8).abs();
        if (dr <= 1 && dc <= 1) && (dr + dc > 0) {
            return true;
        }
    }
    false
}

impl Model<SequenceState> for SequenceHeuristicEvaluator {
    fn evaluate(&self, s: &SequenceState) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_actions(&mut legal);
        let n = legal.len();

        let priors = if n > 0 {
            // Score actions and softmax / normalize priors
            let mut scores: Vec<f32> = legal.iter().map(|a| self.score_action(s, a)).collect();
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = ((*sc - max_score) / 10.0).exp();
                sum_exp += *sc;
            }
            if sum_exp > 0.0 {
                for sc in scores.iter_mut() {
                    *sc /= sum_exp;
                }
                scores
            } else {
                vec![1.0 / n as f32; n]
            }
        } else {
            Vec::new()
        };

        // Compute normalized values across seated players
        let p = s.config.num_players;
        let mut values = vec![0.0f32; p];

        if s.terminated {
            if let Some(win) = s.winner_team {
                let win_val = 1.0;
                let lose_val = if s.config.num_teams == 3 { -0.5 } else { -1.0 };
                for (seat, val) in values.iter_mut().enumerate() {
                    let t = s.config.player_team(seat);
                    *val = if t == win { win_val } else { lose_val };
                }
            }
        } else {
            let mut team_scores = [0.0f32; 3];
            for (t, score) in team_scores.iter_mut().enumerate().take(s.config.num_teams) {
                *score = self.team_score(s, t as u8);
            }
            let avg_score: f32 =
                team_scores[0..s.config.num_teams].iter().sum::<f32>() / s.config.num_teams as f32;

            for (seat, val) in values.iter_mut().enumerate() {
                let t = s.config.player_team(seat) as usize;
                let diff = team_scores[t] - avg_score;
                *val = (diff / 100.0).tanh(); // Normalize into (-1, 1)
            }
        }

        Evaluation { priors, values }
    }
}
