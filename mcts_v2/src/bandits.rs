use std::fmt::Display;

use lib_v2::{utils::Bounds, MctsProblem};
use rand::seq::IteratorRandom;

use crate::forest::Node;

pub trait Bandit<S, A, O> {
  // state is an argument to allow agent/state specific bandit policies
  fn select(&self, state: &S, node: &Node<A, O>, bounds: &Bounds) -> A;
}

pub struct UniformlyRandomBandit;

impl<S, A: Clone, O> Bandit<S, A, O> for UniformlyRandomBandit {
  fn select(&self, _state: &S, node: &Node<A, O>, _bounds: &Bounds) -> A {
    node
      .actions
      .keys()
      .choose(&mut rand::thread_rng())
      .map(|k| k.clone())
      .unwrap()
  }
}

pub struct UctBandit(pub f32);

impl<S, A: Clone + Display, O> Bandit<S, A, O> for UctBandit {
  fn select(&self, _state: &S, node: &Node<A, O>, bounds: &Bounds) -> A {
    println!("bandit start");
    let ln_n = (node.select_count() as f32).ln();
    let mut best_s = f32::MIN;
    let mut best_a = None;
    for (a, data) in node.actions.iter() {
      let n = data.select_count();
      if n == 0 {
        return a.clone();
      }
      let exploration_score = (ln_n / n as f32).sqrt();
      let v = data.value();
      let nv = bounds.normalise(v);
      let score = nv + self.0 * exploration_score;
      println!("a: {a},  score: {score}, v: {v}, nv: {nv}, es: {exploration_score}");
      if score > best_s {
        best_s = score;
        best_a = Some(a);
      }
    }
    best_a.unwrap().clone()
  }
}
