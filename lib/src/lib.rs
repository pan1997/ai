
trait MPOMDP {
  type Agent: Copy;
  type State;
  type Action: PartialEq;
  type Observation: PartialEq;

  fn discount() -> f32 {
    1.0
  }
}


trait State {
  type Agent;
  type Action;
  type Observation;

  fn is_terminal(&self) -> bool;
  fn current_agent(&self) -> Option<Self::Agent>;
  fn legal_actions(&self) -> Vec<Self::Action>;
  fn apply_action(&mut self, action: &Self::Action) -> Vec<(Self::Agent, Self::Observation)>;
}
