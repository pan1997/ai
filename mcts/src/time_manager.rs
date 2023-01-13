use std::{
  fmt::Display,
  time::{Duration, Instant},
};

use lib::{BeliefState, State, MPOMDP};

use crate::{tree::Node, Search, TreeExpansionBlock, TreePolicy};

pub struct Limit {
  // max number of samples
  iteration_limit: Option<u32>,
  // max iteration
  duration_limit: Option<Duration>,
  // granularity
  granularity: u32,
}

impl Limit {
  pub fn iterations(l: u32, granularity: u32) -> Self {
    Limit {
      iteration_limit: Some(l),
      duration_limit: None,
      granularity,
    }
  }

  pub fn time(d: Duration, granularity: u32) -> Self {
    Limit {
      iteration_limit: None,
      duration_limit: Some(d),
      granularity,
    }
  }

  pub fn start<'a, P, T, E>(
    &self,
    search: &Search<'a, P, T, E>,
    belief_state: &P::BeliefState,
    trees: Vec<&Node<P::Action, P::Observation>>,
  ) where
    P: MPOMDP,
    T: TreePolicy<P::State>,
    E: TreeExpansionBlock<P::State>,
    P::Observation: Display,
  {
    let current_agent_ix: usize = belief_state.sample_state().current_agent().unwrap().into();
    let iter_count = self.granularity / search.block_size;
    let start_select_count = trees[current_agent_ix].select_count();
    let start_instant = Instant::now();
    loop {
      for _ in 0..iter_count {
        search.one_block(belief_state, trees.clone());
      }
      let time_millis = start_instant.elapsed().as_millis();
      let mut pv = vec![];
      trees[current_agent_ix].pv(&mut pv, 10);
      let estimated_value = trees[current_agent_ix].value.mean();
      let elapsed_sel = trees[current_agent_ix].select_count() - start_select_count;

      print!("{estimated_value:.4} {elapsed_sel} [ ");
      for (ob, _) in pv {
        print!("{ob} ");
      }
      println!("] {time_millis}ms");

      if self
        .iteration_limit
        .map(|limit| trees[0].select_count() - start_select_count > limit)
        .unwrap_or(false)
        || self
          .duration_limit
          .map(|limit| start_instant.elapsed() > limit)
          .unwrap_or(false)
      {
        break;
      }
    }
  }
}
