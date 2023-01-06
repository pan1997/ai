use lib::{State, MPOMDP};
use tree::Node;
use util::Bounds;

mod tree;
mod util;

mod select;

struct Search<'a, P: MPOMDP, T> {
  problem: &'a P,
  tree_policy: T,

  horizon: u32,
}

trait TreePolicy<S: State> {
  fn select_action<'a: 'b, 'b>(
    &self,
    state: &S,
    node: &'a Node<S::Action, S::Observation>,
    bounds: &Bounds,
  ) -> &'b S::Action;
}
