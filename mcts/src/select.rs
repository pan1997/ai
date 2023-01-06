use lib::{MPOMDP, State};

use crate::tree::Node;
use crate::Search;

impl<'a, P, T> Search<'a, P, T>
where
  P: MPOMDP,
{
  // one tree for each agent
  fn sample(&self, state: &mut P::State, mut trees: Vec<Node<P::Action, P::Observation>>) {
    for _ in 0..self.horizon {
      if state.is_terminal() {
        break;
      }
      
    }
  }
}

struct SelectStep<Agent, Action, Observation> {
  agent_that_moved: Agent,
  action: Action,
  rewards_and_observations: Vec<(f32, Observation)>,
}
