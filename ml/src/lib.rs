use std::fmt::Display;

use lib::{BeliefState, State, MPOMDP};
use mcts::{
  time_manager::Limit,
  util::{Bounds, MctsPolicy},
  Search, TreeExpansionBlock, TreePolicy,
};

fn playout<
  'a,
  P: MPOMDP,
  T1: TreePolicy<P::State>,
  T2: TreePolicy<P::State>,
  E1: TreeExpansionBlock<P::State>,
  E2: TreeExpansionBlock<P::State>,
>(
  belief_state: &P::BeliefState,
  search: Vec<Search<'a, P, T1, E1>>,
  search_limit: &Limit,
) -> Vec<PlayoutStep<P>>
where
  P::Observation: Display,
{
  // the hidden game state
  let mut state = belief_state.sample_state();
  let p = MctsPolicy {};
  let mut result = vec![];

  while !state.is_terminal() {
    let agent = state.current_agent().unwrap();
    let current_agent_index: usize = agent.into();
    let trees = search[current_agent_index].initialize(&state);
    let trees_ref = trees.iter().collect();
    search_limit.start(&search[current_agent_index], belief_state, trees_ref);
    let action = p.select_action(&state, &trees[current_agent_index], &Bounds {}, false);
    let ro = state.apply_action(action);
    result.push(PlayoutStep {
      action: action.clone(),
      agent,
      rewards_and_observations: ro,
      value: vec![0.0; trees.len()],
    });
  }

  result
}

struct PlayoutStep<P: MPOMDP> {
  action: P::Action,
  agent: P::Agent,
  rewards_and_observations: Vec<(f32, P::Observation)>,
  value: Vec<f32>,
}
