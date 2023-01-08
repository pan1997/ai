pub trait MPOMDP {
  type Agent: Copy + TryFrom<usize> + Into<usize>;
  // this is the hidden state
  type State: State<Agent = Self::Agent, Action = Self::Action, Observation = Self::Observation>;
  type Action: Clone + Ord;
  type Observation: Ord + Clone;
  type BeliefState: BeliefState<State = Self::State, Observation = Self::Observation>;

  fn discount(&self) -> f32 {
    1.0
  }

  fn start_state(&self) -> Self::BeliefState;

  //fn agent_count(&self) -> usize;
}

pub trait State {
  type Agent: TryFrom<usize> + Into<usize>;
  type Action: Ord + Clone;
  type Observation: Ord + Clone;

  fn is_terminal(&self) -> bool;
  fn current_agent(&self) -> Option<Self::Agent>;

  // we assume that the set of legal actions is the same for the current agent
  // in all states corresponding to this state's belief state equivalence class
  // Ideally, the legal actions should be a part of BeliefState, instead of
  // state 

  fn legal_actions(&self) -> Vec<Self::Action>;
  // returns the rewards and observations for each agent
  // agents are assumed to be indexed by their Into<usize> impl
  fn apply_action(&mut self, action: &Self::Action) -> Vec<(f32, Self::Observation)>;
}

pub trait BeliefState {
  type State;
  type Observation;
  fn sample_state(&self) -> Self::State;
  fn update(&mut self, o: &Self::Observation);
}

impl<S: State + Clone> BeliefState for S {
  type State = Self;
  type Observation = S::Action;

  fn sample_state(&self) -> Self::State {
    self.clone()
  }

  fn update(&mut self, o: &Self::Observation) {
    self.apply_action(o);
  }
}
