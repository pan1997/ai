use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Classical Monte Carlo rollout evaluation model.
///
/// Estimates leaf value by simulating `num_rollouts` random playouts up to `max_depth`
/// or terminal state, averaging the observed returns:
///
/// $$V(s) \approx \frac{1}{M} \sum_{m=1}^{M} G^{(m)}$$
pub struct RolloutEvaluator<D> {
    /// Reference dynamics used to generate legal actions and execute transitions.
    pub dynamics: D,
    /// Number of random simulation trajectories sampled per evaluation.
    pub num_rollouts: usize,
    /// Maximum search depth before aborting a rollout trajectory.
    pub max_depth: usize,
}

impl<D> RolloutEvaluator<D> {
    /// Constructs a `RolloutEvaluator` with designated rollout budget and depth cut-off.
    pub fn new(dynamics: D, num_rollouts: usize, max_depth: usize) -> Self {
        Self {
            dynamics,
            num_rollouts,
            max_depth,
        }
    }
}

impl<D> Model<D::State> for RolloutEvaluator<D>
where
    D: AgentDynamics + Sync,
    D::State: Clone,
    D::Reward: AsRef<[f32]>,
{
    fn evaluate(&self, s: &D::State) -> Evaluation {
        let legal_actions = self.dynamics.actions(s);
        let num_actions = legal_actions.len();
        let priors = if num_actions > 0 {
            vec![1.0 / (num_actions as f32); num_actions]
        } else {
            Vec::new()
        };

        if num_actions == 0 {
            return Evaluation {
                priors,
                values: vec![0.0],
            };
        }

        let mut total_return = 0.0;
        let mut rng = 123456789u64;

        for _ in 0..self.num_rollouts {
            let mut current = s.clone();
            let mut depth = 0;

            while depth < self.max_depth {
                let actions = self.dynamics.actions(&current);
                if actions.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let pick = (rng as usize) % actions.len();
                let transition = self.dynamics.step(current, &actions[pick]);
                current = transition.next_state;

                if transition.terminated {
                    let rewards = transition.reward.as_ref();
                    if let Some(&first) = rewards.first() {
                        total_return += first;
                    }
                    break;
                }
                depth += 1;
            }
        }

        let avg_value = total_return / (self.num_rollouts as f32);
        Evaluation {
            priors,
            values: vec![avg_value],
        }
    }
}

