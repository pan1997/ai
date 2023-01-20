use std::{fmt::{Debug, Display}, sync::Arc};

use futures::executor::block_on;
use lib::MctsProblem;
use mcts::{bandits::{Bandit, GreedyBandit}, search::Search, SearchLimit, Expansion};
use rand::{distributions::WeightedIndex, prelude::Distribution};
use serde::{Deserialize, Serialize};

pub fn playout<P: MctsProblem, B: Bandit<P::HiddenState, P::Action, P::Observation>, E: Expansion<P>>(
  problem: Arc<P>,
  b_state: &mut P::BeliefState,
  block_size: u32,
  limit: SearchLimit,
  bandit_policy: B,
  mut horizon: u32,
  node_init: E,
  best_only: bool
) -> Vec<PlayoutStep<P::Agent, P::Action, P::Observation>>
where
  P::HiddenState: Clone + Debug,
  P::Observation: Debug,
  P::Action: Debug,
  P::BeliefState: Clone + Debug,
{
  let mut h_state = problem.sample_h_state(b_state);

  let mut result = vec![];
  while horizon != 0 && !problem.check_terminal(&h_state) {
    //println!("{:?}", h_state);
    //println!("{:?}", b_state);
    horizon -= 1;

    let search = Search::new(
      problem.clone(),
      Arc::new(b_state.clone()),
      block_size,
      limit,
      bandit_policy,
      node_init,
    );
    let mut workers = search.create_workers(1);
    search.start(&mut workers[0]);
    let computed_policy = search.get_policy();
    let selected_action = if best_only {
        computed_policy
        .iter()
        .max_by(|(_, a, _), (_, b, _)| a.total_cmp(b))
        .map(|(action, _, _)| action.clone()).unwrap()
    } else {
        let index = WeightedIndex::new(computed_policy.iter().map(|(_, w, _)| w)).unwrap();
        computed_policy[index.sample(&mut rand::thread_rng())]
        .0
        .clone()
    };
    //println!("playing: {:?}", selected_action);
    let current_agent_ix = problem.agent_to_act(&h_state).into() as usize;
    let rewards_and_observations = problem.apply_action(&mut h_state, &selected_action);
    //println!("observed {:?}", rewards_and_observations);
    problem.belief_update(b_state, &rewards_and_observations[current_agent_ix].1);
    result.push(PlayoutStep {
      current_agent: problem.agent_to_act(&h_state),
      computed_policy,
      selected_action,
      rewards_and_observations,
    });
  }
  result
}

pub fn accumulate_rewards<P: MctsProblem>(
  problem: &P,
  playout: &Vec<PlayoutStep<P::Agent, P::Action, P::Observation>>,
) -> Vec<f32> {
  let mut result = vec![0.0; problem.agents().len()];
  let mut factor = 1.0;
  for step in playout.iter() {
    for ix in 0..result.len() {
      result[ix] += factor * step.rewards_and_observations[ix].0;
    }
    factor = problem.discount();
  }
  result
}

#[derive(Serialize, Deserialize)]
pub struct PlayoutStep<Ag, Ac, O> {
  current_agent: Ag,
  computed_policy: Vec<(Ac, f32, f32)>,
  selected_action: Ac,
  rewards_and_observations: Vec<(f32, O)>,
}

impl<Ag, Ac: Display + Debug, O: Display> Debug for PlayoutStep<Ag, Ac, O> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let rewards: Vec<_> = self.rewards_and_observations.iter().map(|x| x.0).collect();
    write!(
      f,
      "{{\"action\": {}, \"rewards\": {:?}, \"policy\": {:?}}}",
      self.selected_action, rewards, self.computed_policy
    )
  }
}

#[cfg(test)]
mod test {
  use examples::prob2;
  use std::sync::Arc;
  use lib::MctsProblem;
  use mcts::{bandits::Uct, EmptyInit, SearchLimit};

  use super::playout;

  #[test]
  fn t1() {
    let m = Arc::new(prob2());
    let mut start = m.start_state();
    let limit = SearchLimit::new(64);
    let bandit_policy = Uct(1.8);
    let t = playout(m, &mut start, 1, limit, bandit_policy, 20, EmptyInit, false);
    println!("{:?}", t);
  }
}
