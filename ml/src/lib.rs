use std::{fmt::{Debug, Display}, sync::Arc};

use futures::executor::block_on;
use lib::MctsProblem;
use mcts::{bandits::Bandit, search::Search, SearchLimit, Expansion};
use rand::{distributions::WeightedIndex, prelude::Distribution};
use serde::{Deserialize, Serialize};

fn playout<P: MctsProblem, B: Bandit<P::HiddenState, P::Action, P::Observation>, E: Expansion<P>>(
  problem: Arc<P>,
  b_state: Arc<P::BeliefState>,
  block_size: u32,
  limit: SearchLimit,
  bandit_policy: B,
  mut horizon: u32,
  node_init: E,
) -> Vec<PlayoutStep<P::Agent, P::Action, P::Observation>>
where
  P::HiddenState: Clone,
{
  let mut h_state = problem.sample_h_state(&b_state);

  let mut result = vec![];
  while horizon != 0 && !problem.check_terminal(&h_state) {
    horizon -= 1;

    let search = Search::new(
      problem.clone(),
      b_state.clone(),
      block_size,
      limit,
      bandit_policy,
      node_init,
    );
    let mut workers = search.create_workers(1);
    search.start(&mut workers[0]);
    let computed_policy = search.get_policy();
    let index = WeightedIndex::new(computed_policy.iter().map(|(_, w)| w)).unwrap();
    let selected_action = computed_policy[index.sample(&mut rand::thread_rng())]
      .0
      .clone();
    let rewards_and_observations = problem.apply_action(&mut h_state, &selected_action);
    result.push(PlayoutStep {
      current_agent: problem.agent_to_act(&h_state),
      computed_policy,
      selected_action,
      rewards_and_observations,
    });
  }
  result
}

fn accumulate_rewards<P: MctsProblem>(
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
struct PlayoutStep<Ag, Ac, O> {
  current_agent: Ag,
  computed_policy: Vec<(Ac, f32)>,
  selected_action: Ac,
  rewards_and_observations: Vec<(f32, O)>,
}

impl<Ag, Ac: Display, O: Display> Debug for PlayoutStep<Ag, Ac, O> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let rewards: Vec<_> = self.rewards_and_observations.iter().map(|x| x.0).collect();
    write!(
      f,
      "{{\"action\": {}, \"rewards\": {:?}}}",
      self.selected_action, rewards
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
    let start = Arc::new(m.start_state());
    let limit = SearchLimit::new(64);
    let bandit_policy = Uct(1.8);
    let t = playout(m, start, 1, limit, bandit_policy, 20, EmptyInit);
    println!("{:?}", t);
  }
}
