use std::cell::Cell;

use graphviz_rust::print;
use lib::State;
use rand::seq::IteratorRandom;

use crate::{tree::Node, TreeExpansion, TreePolicy};

#[derive(Clone)]
pub struct Bounds {}

#[derive(Debug)]
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
    let new_s = old_s + (v - old_s) / (new_n as f32);
    self.stats.replace((new_s, new_n));
  }
}

pub struct RandomTreePolicy;
pub struct UctTreePolicy(pub f32);

impl<S: State> TreePolicy<S> for RandomTreePolicy {
  fn select_action<'a: 'b, 'b>(
    &self,
    _state: &S,
    node: &'a Node<S::Action, S::Observation>,
    _bounds: &Bounds,
    increment_count: bool,
  ) -> &'b S::Action {
    node
      .actions
      .iter()
      .choose(&mut rand::thread_rng())
      .map(|(k, v)| {
        if increment_count {
          v.increment_select_count();
        }
        k
      })
      .unwrap()
  }
}

impl<S: State> TreePolicy<S> for UctTreePolicy {
  fn select_action<'a: 'b, 'b>(
    &self,
    _state: &S,
    node: &'a Node<S::Action, S::Observation>,
    _bounds: &Bounds,
    increment_count: bool,
  ) -> &'b S::Action {
    let ln_N = (node.select_count() as f32).ln();
    let mut best_a = None;
    let mut best_data = None;
    let mut best_score = f32::MIN;
    for (a, data) in node.actions.iter() {
      if data.select_count() == 0 {
        if increment_count {
          data.increment_select_count();
        }
        return a;
      }
      let exploration_score = (ln_N / data.select_count() as f32).sqrt();
      let score = data.action_value() + self.0 * exploration_score;
      if score > best_score {
        best_score = score;
        best_a = Some(a);
        best_data = Some(data);
      }
    }
    if increment_count {
      best_data.unwrap().increment_select_count();
    }
    best_a.unwrap()
  }
}

pub struct EmptyExpansion;

impl<S: State> TreeExpansion<S> for EmptyExpansion {
  fn create_node_and_estimate_value<'a>(
    &self,
    // parent nodes
    nodes: &Vec<&Node<S::Action, S::Observation>>,

    // the last rewards and observations
    rewards_and_observations: &Vec<(f32, S::Observation)>,
    new_state: &S,
  ) -> Vec<f32> {
    let current_agent_index: usize = if new_state.is_terminal() {
      usize::MAX
    } else {
      <S::Agent as Into<usize>>::into(new_state.current_agent().unwrap())
    };

    for ix in 0..nodes.len() {
      if nodes[ix]
        .next_node(&rewards_and_observations[ix].1)
        .is_none()
      {
        let actions = if ix == current_agent_index {
          new_state.legal_actions()
        } else {
          vec![]
        };
        nodes[ix].create_new_node(rewards_and_observations[ix].1.clone(), actions);
      }
    }
    vec![0.0; nodes.len()]
  }
}
