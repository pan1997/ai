use std::{
  collections::BTreeMap,
  fmt::{Debug, Display},
};

use lib::{BeliefState as BeliefState_, State as State_, MPOMDP};
use rand::{distributions::WeightedIndex, prelude::Distribution};

type Action = usize;

#[derive(Clone, Copy, Debug)]
struct Agent;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Observation {
  id: usize,
  action: Action,
}

struct StateDef {
  actions: BTreeMap<Action, ActionResult>,
}

struct ActionResult {
  weights: Vec<f32>,
  next_state_id: Vec<usize>,
  observation: Vec<usize>,
  reward: Vec<f32>,
}

struct StaticPOMDP {
  action_count: usize,
  observation_count: usize,
  state_count: usize,
  states: Vec<StateDef>,
  starting_probabilities: Vec<f32>,
  discount: f32,
}
struct State<'a> {
  id: usize,
  pomdp: &'a StaticPOMDP,
}

impl StaticPOMDP {
  fn new(s: usize, a: usize, o: usize, starting_probabilities: Vec<f32>, discount: f32) -> Self {
    let mut result = StaticPOMDP {
      action_count: a,
      observation_count: o,
      state_count: s,
      states: Vec::with_capacity(s),
      starting_probabilities,
      discount,
    };
    for _ in 0..s {
      result.states.push(StateDef {
        actions: BTreeMap::new(),
      });
    }
    result
  }

  fn add_transition(&mut self, si: usize, a: Action, sj: usize, o: usize, r: f32, w: f32) {
    if !self.states[si].actions.contains_key(&a) {
      self.states[si].actions.insert(a, ActionResult::new());
    }
    let ar = self.states[si].actions.get_mut(&a).unwrap();
    ar.weights.push(w);
    ar.next_state_id.push(sj);
    ar.observation.push(o);
    ar.reward.push(r);
  }
}

impl<'a> MPOMDP for &'a StaticPOMDP {
  type Action = Action;
  type Agent = Agent;
  type Observation = Observation;
  type State = State<'a>;
  type BeliefState = BeliefState<'a>;
  fn discount(&self) -> f32 {
    self.discount
  }

  fn start_state(&self) -> Self::BeliefState {
    BeliefState {
      prob_dist: self.starting_probabilities.clone(),
      pomdp: self,
    }
  }
}

impl<'a> State_ for State<'a> {
  type Action = Action;
  // only one agent
  type Agent = Agent;
  type Observation = Observation;

  fn is_terminal(&self) -> bool {
    self.legal_actions().is_empty()
  }

  fn apply_action(&mut self, action: &Self::Action) -> Vec<(f32, Self::Observation)> {
    let action_result = &self.pomdp.states[self.id].actions[action];
    let wi = WeightedIndex::new(&action_result.weights).unwrap();
    let index = wi.sample(&mut rand::thread_rng());
    self.id = action_result.next_state_id[index];
    vec![(
      action_result.reward[index],
      Observation {
        id: action_result.observation[index],
        action: *action,
      },
    )]
  }

  fn current_agent(&self) -> Option<Self::Agent> {
    if self.is_terminal() {
      None
    } else {
      Some(Agent)
    }
  }

  fn legal_actions(&self) -> Vec<Self::Action> {
    self.pomdp.states[self.id]
      .actions
      .keys()
      .map(|x| *x)
      .collect()
  }
}

impl ActionResult {
  fn new() -> Self {
    ActionResult {
      weights: vec![],
      next_state_id: vec![],
      observation: vec![],
      reward: vec![],
    }
  }
}

impl Into<usize> for Agent {
  fn into(self) -> usize {
    0
  }
}

impl TryFrom<usize> for Agent {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    match value {
      0 => Ok(Agent),
      _ => Err(()),
    }
  }
}

impl Display for Observation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "O({}, a:{})", self.id, self.action)
  }
}

struct BeliefState<'a> {
  prob_dist: Vec<f32>,
  pomdp: &'a StaticPOMDP,
}

impl<'a> BeliefState_ for BeliefState<'a> {
  type Observation = Observation;
  type State = State<'a>;
  fn sample_state(&self) -> Self::State {
    let wi = WeightedIndex::new(&self.prob_dist).unwrap();
    let si = wi.sample(&mut rand::thread_rng());
    State {
      id: si,
      pomdp: self.pomdp,
    }
  }
  fn update(&mut self, o: &Self::Observation) {
    let mut new_dist = vec![0.0; self.prob_dist.len()];
    for s_i in 0..self.pomdp.state_count {
      if self.pomdp.states[s_i].actions.contains_key(&o.action) {
        let transitions = &self.pomdp.states[s_i].actions[&o.action];
        let mut total_w = 0.0;
        for ix in 0..transitions.weights.len() {
          if transitions.observation[ix] == o.id {
            total_w += transitions.weights[ix];
          }
        }

        for ix in 0..transitions.weights.len() {
          if transitions.observation[ix] == o.id {
            new_dist[transitions.next_state_id[ix]] +=
              self.prob_dist[s_i] * transitions.weights[ix] / total_w;
          }
        }
      }
    }

    self.prob_dist = new_dist;
  }
}

#[cfg(test)]
mod test {
  use std::fs::File;

  use lib::{BeliefState, MPOMDP};
  use mcts::{
    tree::{render::save_tree, Node},
    util::{EmptyExpansion, RandomTreePolicy},
    Search,
  };

  use crate::{Agent, StaticPOMDP};

  fn prob1() -> StaticPOMDP {
    let mut m = StaticPOMDP::new(10, 5, 5, vec![0.0; 10], 1.0);
    m.starting_probabilities[0] = 0.5;
    m.starting_probabilities[5] = 0.5;
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

  #[test]
  fn t1() {
    let p = &prob1();
    let s = Search::new(&p, RandomTreePolicy, EmptyExpansion, u32::MAX, true);
    let n = Node::new(&[1, 2]);
    let b_state = p.start_state();
    for _ in 0..100 {
      s.once(&mut b_state.sample_state(), vec![&n]);
    }
    let file = File::create("prob1.dot").unwrap();
    save_tree(&n, file, 10, 3);
  }
}
