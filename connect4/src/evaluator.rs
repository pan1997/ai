//! Baseline evaluation models for Connect 4 MCTS planning.

use crate::dynamics::Connect4Dynamics;
use crate::game::Connect4State;
use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Baseline evaluation model assigning uniform prior probability across non-full columns.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl<const R: usize, const C: usize> Model<Connect4State<R, C>> for UniformEvaluator {
    fn evaluate(&self, s: &Connect4State<R, C>) -> Evaluation {
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
pub struct RolloutEvaluator<const R: usize = 6, const C: usize = 7> {
    pub num_rollouts: usize,
    pub max_depth: usize,
}

impl<const R: usize, const C: usize> RolloutEvaluator<R, C> {
    pub fn new(num_rollouts: usize, max_depth: usize) -> Self {
        Self {
            num_rollouts,
            max_depth,
        }
    }
}

impl<const R: usize, const C: usize> Default for RolloutEvaluator<R, C> {
    fn default() -> Self {
        Self::new(5, 20)
    }
}

impl<const R: usize, const C: usize> Model<Connect4State<R, C>> for RolloutEvaluator<R, C> {
    fn evaluate(&self, s: &Connect4State<R, C>) -> Evaluation {
        let env = Connect4Dynamics::<R, C>;
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
        let mut total_red = 0.0;
        let mut total_yellow = 0.0;
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
                    total_red += outcome.reward[0];
                    total_yellow += outcome.reward[1];
                    break;
                }
                depth += 1;
            }
        }

        let m = self.num_rollouts.max(1) as f32;
        Evaluation {
            priors,
            values: vec![total_red / m, total_yellow / m],
        }
    }
}
