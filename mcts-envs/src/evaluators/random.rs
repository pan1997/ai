use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Baseline evaluation model that assigns uniform prior probability across legal actions and zero value.
///
/// Useful for pure exploration, debugging tree growth, and benchmarking without neural networks:
///
/// $$P(s, a) = \frac{1}{|\mathcal{A}(s)|}, \quad V(s) = [0.0, \dots, 0.0]^\top$$
pub struct UniformRandomModel<D> {
    /// Reference environment dynamics used to query legal actions.
    pub dynamics: D,
    /// Number of players participating in the environment.
    pub num_players: usize,
}

impl<D> UniformRandomModel<D> {
    /// Constructs a `UniformRandomModel` with `dynamics` for `num_players`.
    pub fn new(dynamics: D, num_players: usize) -> Self {
        Self {
            dynamics,
            num_players,
        }
    }
}

impl<D> Model<D::State> for UniformRandomModel<D>
where
    D: AgentDynamics + Sync,
{
    fn evaluate(&self, s: &D::State) -> Evaluation {
        let mut legal = Vec::new();
        self.dynamics.actions(s, &mut legal);
        let n = legal.len();
        let priors = if n > 0 {
            vec![1.0 / (n as f32); n]
        } else {
            Vec::new()
        };
        Evaluation {
            priors,
            values: vec![0.0; self.num_players],
        }
    }
}
