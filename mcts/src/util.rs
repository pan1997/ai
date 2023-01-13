use std::cell::Cell;

use graphviz_rust::print;
use lib::State;
use rand::{
  distributions::WeightedIndex,
  prelude::Distribution,
  seq::{IteratorRandom, SliceRandom},
};

use crate::{tree::Node, TreeExpansion, TreeExpansionBlock, TreePolicy};

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

pub struct GreedyPolicy;
pub struct MctsPolicy;

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
    let mut all_actions: Vec<_> = node.actions.iter().collect();
    all_actions.shuffle(&mut rand::thread_rng());
    for (a, data) in all_actions {
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

impl<S: State> TreePolicy<S> for GreedyPolicy {
  fn select_action<'a: 'b, 'b>(
    &self,
    _state: &S,
    node: &'a Node<S::Action, S::Observation>,
    _bounds: &Bounds,
    _increment_count: bool,
  ) -> &'b S::Action {
    let mut best_a = None;
    let mut best_score = 0;
    for (a, data) in node.actions.iter() {
      let score = data.select_count();
      if score > best_score {
        best_score = score;
        best_a = Some(a);
      }
    }
    best_a.unwrap()
  }
}

impl<S: State> TreePolicy<S> for MctsPolicy {
  fn select_action<'a: 'b, 'b>(
    &self,
    _state: &S,
    node: &'a Node<S::Action, S::Observation>,
    _bounds: &Bounds,
    _increment_count: bool,
  ) -> &'b S::Action {
    let mut actions = vec![];
    let mut w = vec![];
    for (a, data) in node.actions.iter() {
      actions.push(a);
      w.push(data.select_count());
    }
    let ix = WeightedIndex::new(w)
      .unwrap()
      .sample(&mut rand::thread_rng());
    &actions[ix]
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

impl<S: State, E: TreeExpansion<S>> TreeExpansionBlock<S> for E {
  fn create_nodes_and_estimate_values<'a>(
    &self,
    // parent nodes
    nodes_slice: &[Vec<&Node<<S as State>::Action, <S as State>::Observation>>],

    // the last rewards and observations
    rewards_and_observations_slice: &[Vec<(f32, <S as State>::Observation)>],
    new_state_slice: &[S],
  ) -> Vec<Vec<f32>> {
    (0..nodes_slice.len())
      .map(|ix| {
        self.create_node_and_estimate_value(
          &nodes_slice[ix],
          &rewards_and_observations_slice[ix],
          &new_state_slice[ix],
        )
      })
      .collect()
  }
}
