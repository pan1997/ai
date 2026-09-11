use mcts_traits::{AgentDynamics, Evaluation, Model};

pub struct UniformRandomModel<D> {
    pub dynamics: D,
    pub num_players: usize,
}

impl<D> UniformRandomModel<D> {
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
        let legal = self.dynamics.actions(s);
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

