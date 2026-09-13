//! Leaf state evaluation models and heuristic evaluators for 2048 MCTS planning.

use crate::dynamics::Tzf8Dynamics;
use crate::game::Tzf8State;
use mcts_traits::{AgentDynamics, Evaluation, Model};
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// Baseline evaluation model assigning uniform prior probability across legal moves.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl Model<Tzf8State> for UniformEvaluator {
    fn evaluate(&self, s: &Tzf8State) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_directions(&mut legal);
        let n = legal.len();
        let priors = if n > 0 {
            vec![1.0 / (n as f32); n]
        } else {
            Vec::new()
        };
        Evaluation::scalar(priors, 0.0)
    }
}

/// Classic 2048 snake-matrix and monotonicity corner heuristic evaluator.
///
/// Evaluates board quality based on:
/// 1. Snake weight pattern favoring monotonic tile cascades towards any of the 4 corners
///    (evaluated across all 8 rotational and reflective symmetries).
/// 2. Free empty cell bonus, rewarding boards with high flexibility and survival probability.
/// 3. Adjacent tile merge bonus, rewarding immediate fusion potential.
/// 4. Smoothness penalty, penalizing large disparities between adjacent tiles.
#[derive(Debug, Clone, Copy)]
pub struct CornerHeuristicEvaluator {
    /// Weight applied to empty cells (default 250.0).
    pub empty_weight: f32,
    /// Weight applied to adjacent equal mergeable tiles (default 1.0).
    pub merge_weight: f32,
    /// Penalty weight applied to log2 difference between adjacent tiles (default 2.0).
    pub smoothness_weight: f32,
    /// Weight applied to snake matrix score (default 1.0).
    pub snake_weight: f32,
}

impl Default for CornerHeuristicEvaluator {
    fn default() -> Self {
        Self {
            empty_weight: 250.0,
            merge_weight: 1.0,
            smoothness_weight: 2.0,
            snake_weight: 1.0,
        }
    }
}

const BASE_SNAKE_WEIGHTS: [[f32; 4]; 4] = [
    [16.0, 15.0, 14.0, 13.0],
    [9.0, 10.0, 11.0, 12.0],
    [8.0, 7.0, 6.0, 5.0],
    [1.0, 2.0, 3.0, 4.0],
];

impl CornerHeuristicEvaluator {
    /// Creates a new corner heuristic evaluator with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rotates a $4 \times 4$ weight matrix 90 degrees clockwise.
    const fn rotate_90(w: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut out = [[0.0; 4]; 4];
        let mut r = 0;
        while r < 4 {
            let mut c = 0;
            while c < 4 {
                out[c][3 - r] = w[r][c];
                c += 1;
            }
            r += 1;
        }
        out
    }

    /// Horizontally flips a $4 \times 4$ weight matrix.
    const fn flip_h(w: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut out = [[0.0; 4]; 4];
        let mut r = 0;
        while r < 4 {
            let mut c = 0;
            while c < 4 {
                out[r][3 - c] = w[r][c];
                c += 1;
            }
            r += 1;
        }
        out
    }

    /// Evaluates the static scalar heuristic score of `state`.
    pub fn evaluate_state(&self, state: &Tzf8State) -> f32 {
        if !state.ongoing || !state.can_move() {
            return -100_000.0;
        }

        // 1. Compute snake score across 8 symmetries (4 rotations x 2 flips)
        let mut w0 = BASE_SNAKE_WEIGHTS;
        let mut max_snake = f32::NEG_INFINITY;

        for _ in 0..4 {
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let w_flip = Self::flip_h(&w0);

            for r in 0..4 {
                for c in 0..4 {
                    let tile = state.board[r][c] as f32;
                    s1 += tile * w0[r][c];
                    s2 += tile * w_flip[r][c];
                }
            }

            if s1 > max_snake {
                max_snake = s1;
            }
            if s2 > max_snake {
                max_snake = s2;
            }

            w0 = Self::rotate_90(&w0);
        }

        // 2. Empty cell count
        let empty_count = state.count_empty_cells() as f32;
        let empty_score = empty_count * self.empty_weight;

        // 3. Smoothness and merge bonus
        let mut smoothness_penalty = 0.0;
        let mut merge_bonus = 0.0;

        for r in 0..4 {
            for c in 0..4 {
                let v = state.board[r][c];
                if v == 0 {
                    continue;
                }
                let rank = (v as f32).log2();

                // Right neighbor
                if c + 1 < 4 {
                    let right = state.board[r][c + 1];
                    if right > 0 {
                        let right_rank = (right as f32).log2();
                        smoothness_penalty += (rank - right_rank).abs();
                        if v == right {
                            merge_bonus += v as f32;
                        }
                    }
                }

                // Down neighbor
                if r + 1 < 4 {
                    let down = state.board[r + 1][c];
                    if down > 0 {
                        let down_rank = (down as f32).log2();
                        smoothness_penalty += (rank - down_rank).abs();
                        if v == down {
                            merge_bonus += v as f32;
                        }
                    }
                }
            }
        }

        self.snake_weight * max_snake + empty_score + self.merge_weight * merge_bonus
            - self.smoothness_weight * smoothness_penalty
    }
}

impl Model<Tzf8State> for CornerHeuristicEvaluator {
    fn evaluate(&self, s: &Tzf8State) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_directions(&mut legal);
        let n = legal.len();
        if n == 0 {
            return Evaluation::scalar(Vec::new(), -100_000.0);
        }

        let priors = vec![1.0 / (n as f32); n];
        let val = self.evaluate_state(s);
        Evaluation::scalar(priors, val)
    }
}

/// Rollout evaluator estimating 2048 state value via random simulation playouts.
#[derive(Debug, Clone, Copy)]
pub struct RolloutEvaluator {
    /// Number of random simulation sweeps per leaf evaluation.
    pub num_rollouts: usize,
    /// Maximum search depth per rollout.
    pub max_depth: usize,
}

impl RolloutEvaluator {
    /// Creates a new `RolloutEvaluator` with specified parameters.
    pub fn new(num_rollouts: usize, max_depth: usize) -> Self {
        Self {
            num_rollouts,
            max_depth,
        }
    }
}

impl Default for RolloutEvaluator {
    fn default() -> Self {
        Self::new(5, 25)
    }
}

impl Model<Tzf8State> for RolloutEvaluator {
    fn evaluate(&self, s: &Tzf8State) -> Evaluation {
        let dynamics = Tzf8Dynamics::new();
        let mut legal = Vec::new();
        s.legal_directions(&mut legal);
        let n = legal.len();
        if n == 0 {
            return Evaluation::scalar(Vec::new(), 0.0);
        }

        let priors = vec![1.0 / (n as f32); n];
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut total_score_gain = 0.0;

        for _ in 0..self.num_rollouts {
            let mut sim_state = s.clone();
            let mut rollout_gain = 0.0;

            for _ in 0..self.max_depth {
                legal.clear();
                dynamics.actions(&sim_state, &mut legal);
                if legal.is_empty() {
                    break;
                }
                let act = *legal.choose(&mut rng).unwrap();
                let outcome = dynamics.step(&mut sim_state, &act);
                rollout_gain += outcome.reward[0];
                if outcome.terminated {
                    break;
                }
            }
            total_score_gain += rollout_gain;
        }

        let mean_val = total_score_gain / (self.num_rollouts as f32);
        Evaluation::scalar(priors, mean_val)
    }
}
