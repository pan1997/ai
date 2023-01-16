use forest::{Forest, Node};
use lib::MctsProblem;
use search::Trajectory;

pub mod bandits;
pub mod forest;
pub mod rollout;
pub mod search;

pub trait NodeInit<P: MctsProblem>: Copy {
  // one node per player

  // we assume that the legal actions have been populated by search, and this is responsible for
  // putting in the static policy scores, and providing a value estimate for the states

  fn init_node(
    &self,
    problem: &P,
    state: &P::HiddenState,
    node_of_current_agent: &mut Node<P::Action, P::Observation>,
  ) -> Vec<f32>;

  fn block_init(
    &self,
    problem: &P,
    states: &[P::HiddenState],
    forest: &mut Forest<P::Action, P::Observation>,
    trajectories: &[Trajectory<P::Action>],
  ) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(states.len());
    for ix in 0..states.len() {
      let current_agent_ix = problem.agent_to_act(&states[ix]).into() as usize;
      result.push(self.init_node(
        problem,
        &states[ix],
        forest.node_mut(trajectories[ix].current_[current_agent_ix]),
      ));
    }
    result
  }
}

#[derive(Clone, Copy)]
pub struct SearchLimit {
  node_count: Option<u32>,
}

impl SearchLimit {
  fn more(&self, n: u32) -> bool {
    if self.node_count.map(|l| n > l).unwrap_or(false) {
      return false;
    }
    // more checks

    true
  }

  pub fn new(n: u32) -> Self {
    SearchLimit {
      node_count: Some(n),
    }
  }
}

pub use rollout::EmptyInit;
