use std::fmt::Debug;

/// Result of stepping an environment state with an action.
///
/// Encapsulates the subsequent state, immediate reward signal, and termination flag.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition<S, R> {
    /// The resulting state after taking the action.
    pub next_state: S,
    /// Immediate transition reward received (e.g., scalar return or per-agent reward vector).
    pub reward: R,
    /// Indicates whether the next state is a terminal game/episode state.
    pub terminated: bool,
}

impl<S, R> Transition<S, R> {
    /// Constructs a new `Transition`.
    #[inline]
    pub fn new(next_state: S, reward: R, terminated: bool) -> Self {
        Self {
            next_state,
            reward,
            terminated,
        }
    }
}

/// Minimal single-agent transition dynamics interface for tree planning.
///
/// In MCTS, an agent hypothesizes future trajectories by querying this trait.
/// Associated types have no hardcoded `Send + Sync + Clone` bounds; algorithms
/// that require these bounds declare them on their own methods or structs.
pub trait AgentDynamics {
    /// Internal representation of the environment or planning state.
    type State;
    /// Action representation; must implement `Eq + Debug` for verification and hashing.
    type Action: Eq + Debug;
    /// Reward representation (typically a scalar `f32` or multi-agent array `[f32; N]`).
    type Reward;

    /// Returns the initial or root state for planning.
    fn initial(&self) -> Self::State;

    /// Generates the list of legal actions available in state `s`.
    fn actions(&self, s: &Self::State) -> Vec<Self::Action>;

    /// Transitions state `s` forward given `action`.
    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward>;
}

/// High-throughput batched dynamics for vectorized environments or MuZero neural dynamics.
///
/// Allows amortizing simulation overhead across multiple states and actions simultaneously.
pub trait BatchedAgentDynamics: AgentDynamics {
    /// Steps a batch of states and corresponding actions forward in lockstep.
    ///
    /// Writes resulting transitions into `out_transitions`.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `states.len() != actions.len()`.
    fn step_batch(
        &self,
        states: &[Self::State],
        actions: &[Self::Action],
        out_transitions: &mut Vec<Transition<Self::State, Self::Reward>>,
    );
}

/// Reference sequential implementation of [`BatchedAgentDynamics::step_batch`].
///
/// Iterates over matching pairs of `states` and `actions`, stepping each through `dynamics.step`.
///
/// # Panics
///
/// Panics if `states.len() != actions.len()`.
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
