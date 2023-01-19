use forest::{Forest, Node};
use lib::MctsProblem;
use search::Trajectory;

pub mod bandits;
pub mod forest;
pub mod rollout;
pub mod search;

pub trait Expansion<P>: Copy
where
  P: MctsProblem,
{
  // scores per agent, and then static policy per action
  fn expand(&self, p: &P, s: &P::HiddenState) -> (Vec<f32>, Vec<(P::Action, f32)>);

  fn block_expand(
    &self,
    p: &P,
    states: &[P::HiddenState],
  ) -> (Vec<Vec<f32>>, Vec<Vec<(P::Action, f32)>>) {
    let mut r1 = vec![];
    let mut r2 = vec![];
    states.iter().map(|s| self.expand(p, s)).for_each(|(v, p)| {
      r1.push(v);
      r2.push(p);
    });
    (r1, r2)
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
