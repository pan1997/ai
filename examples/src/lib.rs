use std::{collections::BTreeMap, fmt::Display};

use lib::MctsProblem;
use rand::distributions::{Distribution, WeightedIndex};

type Action = usize;
#[derive(Copy, Clone, Debug)]
pub struct Agent;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Observation {
  id: usize,
  action: Action,
}

struct ActionDef {
  weights: Vec<f32>,
  next_state_id: Vec<usize>,
  observation_id: Vec<usize>,
  reward: Vec<f32>,
}

struct StateDef {
  outgoing_actions: BTreeMap<usize, ActionDef>,
}

#[derive(Clone, Debug)]
pub struct BeliefState {
  state_probs: Vec<f32>,
}

pub struct StaticPOMDP {
  state_count: usize,
  observation_count: usize,
  action_count: usize,
  discount: f32,
  states: Vec<StateDef>,
  start_state: BeliefState,
}

impl StaticPOMDP {
  fn new(s: usize, a: usize, o: usize, state_probs: Vec<f32>, discount: f32) -> Self {
    let mut result = StaticPOMDP {
      action_count: a,
      observation_count: o,
      state_count: s,
      states: Vec::with_capacity(s),
      start_state: BeliefState { state_probs },
      discount,
    };
    for _ in 0..s {
      result.states.push(StateDef {
        outgoing_actions: BTreeMap::new(),
      });
    }
    result
  }

  fn add_transition(&mut self, si: usize, a: Action, sj: usize, o: usize, r: f32, w: f32) {
    assert!(a < self.action_count, "invalid action");
    assert!(o < self.observation_count, "invalid observation");
    if !self.states[si].outgoing_actions.contains_key(&a) {
      self.states[si].outgoing_actions.insert(
        a,
        ActionDef {
          weights: vec![],
          next_state_id: vec![],
          observation_id: vec![],
          reward: vec![],
        },
      );
    }
    let ar = self.states[si].outgoing_actions.get_mut(&a).unwrap();
    ar.weights.push(w);
    ar.next_state_id.push(sj);
    ar.observation_id.push(o);
    ar.reward.push(r);
  }
}

impl MctsProblem for StaticPOMDP {
  type Action = usize;
  type Agent = Agent;
  type BeliefState = BeliefState;
  type HiddenState = usize;
  type Observation = Observation;

  fn apply_action(
    &self,
    h_state: &mut Self::HiddenState,
    action: &Self::Action,
  ) -> Vec<(f32, Self::Observation)> {
    let action_result = &self.states[*h_state].outgoing_actions[action];
    let wi = WeightedIndex::new(&action_result.weights).unwrap();
    let index = wi.sample(&mut rand::thread_rng());
    *h_state = action_result.next_state_id[index];
    vec![(
      action_result.reward[index],
      Observation {
        id: action_result.observation_id[index],
        action: *action,
      },
    )]
  }

  fn belief_update(&self, b_state: &mut Self::BeliefState, obs: &Self::Observation) {
    let mut new_dist = vec![0.0; b_state.state_probs.len()];
    for s_i in 0..self.state_count {
      if self.states[s_i].outgoing_actions.contains_key(&obs.action) {
        let transitions = &self.states[s_i].outgoing_actions[&obs.action];
        let mut total_w = 0.0;
        for ix in 0..transitions.weights.len() {
          if transitions.observation_id[ix] == obs.id {
            total_w += transitions.weights[ix];
          }
        }

        for ix in 0..transitions.weights.len() {
          if transitions.observation_id[ix] == obs.id {
            new_dist[transitions.next_state_id[ix]] +=
              b_state.state_probs[s_i] * transitions.weights[ix] / total_w;
          }
        }
      }
    }
    b_state.state_probs = new_dist;
  }

  fn legal_actions(&self, h_state: &Self::HiddenState) -> Vec<Self::Action> {
    self.states[*h_state]
      .outgoing_actions
      .keys()
      .map(|x| *x)
      .collect()
  }

  fn sample_h_state(&self, b_state: &Self::BeliefState) -> Self::HiddenState {
    let wi = WeightedIndex::new(&b_state.state_probs).unwrap();
    wi.sample(&mut rand::thread_rng())
  }

  fn check_terminal(&self, h_state: &Self::HiddenState) -> bool {
    self.legal_actions(h_state).is_empty()
  }

  fn start_state(&self) -> Self::BeliefState {
    self.start_state.clone()
  }

  fn agents(&self) -> Vec<Self::Agent> {
    vec![Agent]
  }

  fn agent_to_act(&self, _h_state: &Self::HiddenState) -> Self::Agent {
    Agent
  }
  fn discount(&self) -> f32 {
    self.discount
  }
}

impl From<Agent> for u8 {
  fn from(_: Agent) -> Self {
    0
  }
}

impl Display for Observation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "O({}, a:{})", self.id, self.action)
  }
}

pub fn prob1() -> StaticPOMDP {
  let mut s_prob = vec![0.0; 10];
  s_prob[0] = 0.5;
  s_prob[5] = 0.5;
  let mut m = StaticPOMDP::new(10, 5, 5, s_prob, 1.0);
  m.add_transition(0, 1, 1, 0, 0.0, 1.0);
  m.add_transition(0, 2, 2, 0, 0.5, 1.0);
  m.add_transition(1, 3, 3, 1, -1.0, 1.0);

  m.add_transition(1, 4, 4, 2, 1.0, 1.0);
  m.add_transition(5, 1, 6, 0, 0.0, 1.0);
  m.add_transition(5, 2, 7, 0, 0.5, 1.0);
  m.add_transition(6, 3, 8, 3, 1.0, 1.0);
  m.add_transition(6, 4, 9, 4, -1.0, 1.0);
  m
}

pub fn prob2() -> StaticPOMDP {
  let mut m = StaticPOMDP::new(3, 2, 3, vec![1.0, 0.0, 0.0], 1.0);

  m.add_transition(0, 0, 0, 0, 0.0, 0.5);
  m.add_transition(0, 0, 2, 2, 0.0, 0.5);
  m.add_transition(0, 1, 2, 2, 0.0, 1.0);

  m.add_transition(1, 0, 0, 0, 5.0, 0.7);
  m.add_transition(1, 0, 1, 1, 0.0, 0.1);
  m.add_transition(1, 0, 2, 2, 0.0, 0.2);
  m.add_transition(1, 1, 1, 1, 0.0, 0.95);
  m.add_transition(1, 1, 2, 2, 0.0, 0.05);

  m.add_transition(2, 0, 0, 0, 0.0, 0.4);
  m.add_transition(2, 0, 2, 2, 0.0, 0.6);
  m.add_transition(2, 1, 0, 0, -1.0, 0.3);
  m.add_transition(2, 1, 1, 1, 0.0, 0.3);
  m.add_transition(2, 1, 2, 2, 0.0, 0.4);

  m
}
#[cfg(test)]
pub mod tests {
  use std::{fs::File, sync::Arc};

  use mcts::{bandits::Uct, forest::render::save, search::Search, EmptyInit, SearchLimit};

  use super::*;

  #[test]
  fn test1() {
    let problem = Arc::new(prob1());
    let start_state = Arc::new(problem.start_state());
    let limit = SearchLimit::new(1000);
    let search = Search::new(
      problem.clone(),
      start_state.clone(),
      1,
      limit,
      Uct(2.4),
      EmptyInit,
    );
    let mut worker = search.create_workers(1);
    println!("created");
    search.start(&mut worker[0]);
    let forest = search.forest.read().unwrap();
    //println!("{:?}", forest);
    save(&forest, File::create("agent.dot").unwrap(), 0, 3);
  }

  #[test]
  fn test2() {
    let problem = Arc::new(prob2());
    let start_state = Arc::new(problem.start_state());
    let limit = SearchLimit::new(10000);
    let search = Search::new(
      problem.clone(),
      start_state.clone(),
      1,
      limit,
      Uct(1.2),
      EmptyInit,
    );
    let mut worker = search.create_workers(1);
    println!("created");
    search.start(&mut worker[0]);
    let forest = search.forest.read().unwrap();
    //println!("{:?}", forest);
    save(&forest, File::create("agent.dot").unwrap(), 500, 5);
  }
}
