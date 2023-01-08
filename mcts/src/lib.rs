use lib::{State, MPOMDP};
use tree::Node;
use util::Bounds;

pub mod tree;
pub mod util;

mod search;

pub struct Search<'a, P: MPOMDP, T: TreePolicy<P::State>, E> {
  problem: &'a P,
  tree_policy: T,
  tree_expansion: E,

  horizon: u32,
  bounds: Vec<Bounds>,
}

pub trait TreePolicy<S: State> {
  fn select_action<'a: 'b, 'b>(
    &self,
    state: &S,
    node: &'a Node<S::Action, S::Observation>,
    bounds: &Bounds,
  ) -> &'b S::Action;
}

pub trait TreeExpansion<S: State> {
  // create node and return a value estimate
  fn create_node_and_estimate_value<'a>(
    &self,
    // parent nodes
    nodes: &Vec<&Node<S::Action, S::Observation>>,

    // the last rewards and observations
    rewards_and_observations: &Vec<(f32, S::Observation)>,
    new_state: &S,
  ) -> Vec<f32>;
}

impl<'a, P: MPOMDP, T: TreePolicy<P::State>, E> Search<'a, P, T, E> {
  pub fn new(problem: &'a P, tree_policy: T, tree_expansion: E) -> Self {
    Search {
      problem,
      tree_policy,
      tree_expansion,
      horizon: u32::MAX,
      bounds: vec![Bounds {}; 10],
    }
  }
}
