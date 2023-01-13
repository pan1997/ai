use crate::{BeliefState, State as State_, MPOMDP};

// fully observable deterministic multi agent mdp
pub trait FODMMDP {
  type Agent: Copy + TryFrom<usize> + Into<usize>;
  type Action: Clone + Ord;
  type State: Clone + State<Agent = Self::Agent, Action = Self::Action>;

  fn start_state(&self) -> Self::State;

  fn all_agents(&self) -> Vec<Self::Agent>;
}

impl<P> MPOMDP for P
where
  P: FODMMDP,
{
  type Agent = P::Agent;
  type Action = P::Action;
  type Observation = P::Action;
  type State = P::State;
  type BeliefState = P::State;

  fn start_state(&self) -> Self::BeliefState {
    self.start_state()
  }
  fn all_agents(&self) -> Vec<Self::Agent> {
    self.all_agents()
  }
}

pub trait State {
  type Agent: Copy + TryFrom<usize> + Into<usize>;
  type Action: Clone + Ord;

  fn current_agent(&self) -> Option<Self::Agent>;
  fn legal_actions(&self) -> Vec<Self::Action>;
  fn is_terminal(&self) -> bool;
  // returns rewards for each player
  fn apply_action(&mut self, action: &Self::Action) -> Vec<f32>;
}

impl<S: State + Clone> BeliefState for S {
  type Observation = S::Action;
  type State = Self;
  fn sample_state(&self) -> Self::State {
    self.clone()
  }
  fn update(&mut self, o: &Self::Observation) {
    self.apply_action(o);
  }
}

impl<S: State> State_ for S {
  type Action = S::Action;
  type Agent = S::Agent;
  type Observation = S::Action;
  fn apply_action(&mut self, action: &Self::Action) -> Vec<(f32, Self::Observation)> {
    self
      .apply_action(action)
      .into_iter()
      .map(|r| (r, action.clone()))
      .collect()
  }
  fn current_agent(&self) -> Option<Self::Agent> {
    self.current_agent()
  }
  fn is_terminal(&self) -> bool {
    self.is_terminal()
  }
  fn legal_actions(&self) -> Vec<Self::Action> {
    self.legal_actions()
  }
}
