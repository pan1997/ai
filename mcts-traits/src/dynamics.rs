use std::fmt::Debug;

/// Result of stepping a state with an action.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition<S, R> {
    pub next_state: S,
    pub reward: R,
    pub terminated: bool,
}

impl<S, R> Transition<S, R> {
    #[inline]
    pub fn new(next_state: S, reward: R, terminated: bool) -> Self {
        Self {
            next_state,
            reward,
            terminated,
        }
    }
}

/// Minimal single-agent transition dynamics for tree planning.
///
/// ### Design Note:
/// Associated types have no hardcoded `Send + Sync + Clone` bounds.
/// Algorithms that need those bounds declare them on their own methods/structs.
pub trait AgentDynamics {
    type State;
    type Action: Eq + Debug;
    type Reward;

    /// Initial state for search (e.g. root state or initial decision point).
    fn initial(&self) -> Self::State;

    /// Legal actions available in state `s`.
    fn actions(&self, s: &Self::State) -> Vec<Self::Action>;

    /// Transitions the state forward given an action.
    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward>;
}

/// High-throughput batched dynamics for vectorized environments or MuZero neural dynamics.
pub trait BatchedAgentDynamics: AgentDynamics {
    fn step_batch(
        &self,
        states: &[Self::State],
        actions: &[Self::Action],
        out_transitions: &mut Vec<Transition<Self::State, Self::Reward>>,
    );
}

/// Helper function to process batched steps sequentially.
pub fn default_step_batch<D>(
    dynamics: &D,
    states: &[D::State],
    actions: &[D::Action],
    out_transitions: &mut Vec<Transition<D::State, D::Reward>>,
) where
    D: AgentDynamics + ?Sized,
    D::State: Clone,
{
    assert_eq!(
        states.len(),
        actions.len(),
        "default_step_batch: states and actions slice lengths must match"
    );
    out_transitions.clear();
    out_transitions.reserve(states.len());
    for (s, a) in states.iter().zip(actions.iter()) {
        out_transitions.push(dynamics.step(s.clone(), a));
    }
}
