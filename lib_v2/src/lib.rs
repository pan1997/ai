pub mod utils;

pub trait MctsProblem {
  type Agent: Copy + Into<u8>;

  type Action: Clone + Ord;

  type Observation: Clone + Ord;

  type BeliefState;

  type HiddenState;

  fn start_state(&self) -> Self::BeliefState;

  fn sample_h_state(&self, b_state: &Self::BeliefState) -> Self::HiddenState;
  fn belief_update(&self, b_state: &mut Self::BeliefState, obs: &Self::Observation);

  // these two are assumed consistent, that for every h_state that can be sampled from a `b_state`,
  // the agent to act is same
  // every state has one agent that's supposed to move.
  // for terminal states, this can be arbitrary
  fn agent_to_act(&self, h_state: &Self::HiddenState) -> Self::Agent;

  // fn agent_to_act_in_b_state(&self, b_state: &Self::BeliefState) -> Self::Agent;
  fn legal_actions(&self, h_state: &Self::HiddenState) -> Vec<Self::Action>;
  fn apply_action(
    &self,
    h_state: &mut Self::HiddenState,
    action: &Self::Action,
  ) -> Vec<(f32, Self::Observation)>;

  fn check_terminal(&self, h_state: &Self::HiddenState) -> bool;

  fn agents(&self) -> Vec<Self::Agent>;

  // utils
  fn discount(&self) -> f32 {
    1.0
  }

  fn sample_h_state_batched(
    &self,
    b_state: &Self::BeliefState,
    count: usize,
  ) -> Vec<Self::HiddenState> {
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
      result.push(self.sample_h_state(b_state));
    }
    result
  }

  fn apply_action_batched(
    &self,
    h_states: &mut [Self::HiddenState],
    actions: &[Self::Action],
  ) -> Vec<Vec<(f32, Self::Observation)>> {
    let mut result = Vec::with_capacity(h_states.len());
    for (h_state, action) in h_states.iter_mut().zip(actions) {
      result.push(self.apply_action(h_state, action));
    }
    result
  }
}

pub trait FullyObservableDeterministicMctsProblem {
  type Agent: Copy + Into<u8>;
  type Action: Clone + Ord;
  type State: Clone;

  fn agents(&self) -> Vec<Self::Agent>;
  fn start_state(&self) -> Self::State;
  fn agent_to_act(&self, state: &Self::State) -> Self::Agent;
  fn check_terminal(&self, state: &Self::State) -> bool;
  fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;
  fn apply_action(&self, state: &mut Self::State, action: &Self::Action) -> Vec<f32>;
  // util
  fn discount(&self) -> f32 {
    1.0
  }
}

impl<T> MctsProblem for T
where
  T: FullyObservableDeterministicMctsProblem,
{
  type Agent = T::Agent;
  type Action = T::Action;
  type Observation = T::Action;
  type HiddenState = T::State;
  type BeliefState = T::State;
  fn agent_to_act(&self, h_state: &Self::HiddenState) -> Self::Agent {
    self.agent_to_act(h_state)
  }
  fn check_terminal(&self, h_state: &Self::HiddenState) -> bool {
    self.check_terminal(h_state)
  }
  fn discount(&self) -> f32 {
    self.discount()
  }
  fn legal_actions(&self, h_state: &Self::HiddenState) -> Vec<Self::Action> {
    self.legal_actions(h_state)
  }
  fn sample_h_state(&self, b_state: &Self::BeliefState) -> Self::HiddenState {
    b_state.clone()
  }
  fn start_state(&self) -> Self::BeliefState {
    self.start_state()
  }
  fn belief_update(&self, b_state: &mut Self::BeliefState, obs: &Self::Observation) {
    self.apply_action(b_state, obs);
  }
  fn apply_action(
    &self,
    h_state: &mut Self::HiddenState,
    action: &Self::Action,
  ) -> Vec<(f32, Self::Observation)> {
    let rewards = self.apply_action(h_state, action);
    self
      .agents()
      .into_iter()
      .map(|agent| (rewards[agent.into() as usize], action.clone()))
      .collect()
  }
  fn agents(&self) -> Vec<Self::Agent> {
    self.agents()
  }
}
