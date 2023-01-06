use std::cell::Cell;

use lib::State;
use rand::seq::IteratorRandom;

use crate::tree::Node;
use crate::TreePolicy;

pub(crate) struct Bounds {}

pub(crate) struct Average {
  stats: Cell<(f32, u32)>,
}

impl Average {
  pub(crate) fn new() -> Self {
    Average {
      stats: Cell::new((0.0, 0)),
    }
  }

  pub(crate) fn mean(&self) -> f32 {
    self.stats.get().0
  }

  pub(crate) fn sample_count(&self) -> u32 {
    self.stats.get().1
  }

  pub(crate) fn add_sample(&self, v: f32, n: u32) {
    let (old_s, old_n) = self.stats.get();
    let new_n = old_n + n;
    let new_s = old_s + (v - old_s) / new_n as f32;
    self.stats.replace((new_s, new_n));
  }
}

struct RandomTreePolicy;
struct UctTreePolicy(f32);

impl<S: State> TreePolicy<S> for RandomTreePolicy {
  fn select_action<'a: 'b, 'b>(
    &self,
    state: &S,
    node: &'a Node<S::Action, S::Observation>,
    bounds: &Bounds,
  ) -> &'b S::Action {
    node.actions.keys().choose(&mut rand::thread_rng()).unwrap()
  }
}

impl<S: State> TreePolicy<S> for UctTreePolicy {
  fn select_action<'a: 'b, 'b>(
    &self,
    state: &S,
    node: &'a Node<S::Action, S::Observation>,
    bounds: &Bounds,
  ) -> &'b S::Action {
    unimplemented!()
  }
}
