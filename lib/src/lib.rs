pub trait MPOMDP {
  type Agent: Copy + TryFrom<usize> + Into<usize>;
  // this is the hidden state
  type State: State<Agent = Self::Agent, Action = Self::Action, Observation = Self::Observation>;
  type Action: PartialEq;
  type Observation: PartialEq;

  fn discount() -> f32 {
    1.0
  }
}

pub trait State {
  type Agent;
  type Action;
  type Observation;

  fn is_terminal(&self) -> bool;
  fn current_agent(&self) -> Option<Self::Agent>;
  fn legal_actions(&self) -> Vec<Self::Action>;
  // returns the rewards and observations for each agent
  // agents are assumed to be indexed by their Into<usize> impl
  fn apply_action(&mut self, action: &Self::Action) -> Vec<(f32, Self::Observation)>;
}
