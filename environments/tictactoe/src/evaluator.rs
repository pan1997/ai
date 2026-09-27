//! Baseline evaluation models for TicTacToe MCTS planning.

use crate::dynamics::TicTacToeDynamics;
use crate::game::TicTacToeState;
use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Baseline evaluation model assigning uniform prior probability across legal cells.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl Model<TicTacToeState> for UniformEvaluator {
    fn evaluate(&self, s: &TicTacToeState) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_actions(&mut legal);
        let n = legal.len();
        let priors = if n > 0 {
            vec![1.0 / (n as f32); n]
        } else {
            Vec::new()
        };
        Evaluation {
            priors,
            values: vec![0.0, 0.0],
        }
    }
}

/// Rollout evaluator estimating state value via random simulation playouts.
#[derive(Debug, Clone, Copy)]
pub struct RolloutEvaluator {
    /// Number of random simulation playouts per leaf node evaluation.
    pub num_rollouts: usize,
    /// Maximum search depth before truncating the rollout.
    pub max_depth: usize,
}

impl RolloutEvaluator {
    /// Constructs a new `RolloutEvaluator` with the specified rollout count and maximum depth.
    pub fn new(num_rollouts: usize, max_depth: usize) -> Self {
        Self {
            num_rollouts,
            max_depth,
        }
    }
}

impl Default for RolloutEvaluator {
    fn default() -> Self {
        Self::new(10, 10)
    }
}

impl Model<TicTacToeState> for RolloutEvaluator {
    fn evaluate(&self, s: &TicTacToeState) -> Evaluation {
        let env = TicTacToeDynamics;
        let mut legal_actions = Vec::new();
        s.legal_actions(&mut legal_actions);
        let num_actions = legal_actions.len();
        if num_actions == 0 {
            return Evaluation {
                priors: Vec::new(),
                values: vec![0.0, 0.0],
            };
        }

        let priors = vec![1.0 / (num_actions as f32); num_actions];
        let mut total_x = 0.0;
        let mut total_o = 0.0;
        let mut rng = rand::thread_rng();
        let mut actions_buf = Vec::new();

        for _ in 0..self.num_rollouts {
            let mut current = s.clone();
            let mut depth = 0;

            while depth < self.max_depth {
                current.legal_actions(&mut actions_buf);
                if actions_buf.is_empty() {
                    break;
                }
                let pick = rand::Rng::gen_range(&mut rng, 0..actions_buf.len());
                let outcome = env.step(&mut current, &actions_buf[pick]);

                if outcome.terminated {
                    total_x += outcome.reward[0];
                    total_o += outcome.reward[1];
                    break;
                }
                depth += 1;
            }
        }

        let m = self.num_rollouts.max(1) as f32;
        Evaluation {
            priors,
            values: vec![total_x / m, total_o / m],
        }
    }
}
