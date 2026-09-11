use std::fmt::Debug;

/// Outcome of stepping an environment state in-place with an action.
///
/// Encapsulates the immediate reward signal received and the episode termination flag.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepOutcome<R> {
    /// Immediate transition reward received (e.g., scalar return or per-agent reward vector).
    pub reward: R,
    /// Indicates whether the subsequent state is a terminal game/episode state.
    pub terminated: bool,
}

impl<R> StepOutcome<R> {
    /// Constructs a new `StepOutcome`.
    #[inline]
    pub fn new(reward: R, terminated: bool) -> Self {
        Self { reward, terminated }
    }
}

/// Result of stepping an environment state with an action, holding ownership of the next state.
///
/// Primarily used for transition logging, experience replay buffers, or non-destructive simulation.
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

    /// Generates legal actions available in state `s` into `out`.
    ///
    /// Clears or appends to `out` to guarantee zero heap allocations during search tree expansion.
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>);

    /// Transitions state `s` forward in-place given `action`.
    ///
    /// Modifies `s` directly without heap allocation or intermediate cloning, returning the
    /// immediate reward and termination status.
    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward>;
}

/// High-throughput batched dynamics for vectorized environments or MuZero neural dynamics.
///
/// Allows amortizing simulation overhead across multiple states and actions simultaneously.
pub trait BatchedAgentDynamics: AgentDynamics {
    /// Steps a batch of states and corresponding actions forward in lockstep in-place.
    ///
    /// Modifies each state in `states` in-place and writes resulting outcomes into `out_outcomes`.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `states.len() != actions.len()`.
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
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
    states: &mut [D::State],
    actions: &[D::Action],
    out_outcomes: &mut Vec<StepOutcome<D::Reward>>,
) where
    D: AgentDynamics + ?Sized,
{
    assert_eq!(
        states.len(),
        actions.len(),
        "default_step_batch: states and actions slice lengths must match"
    );
    out_outcomes.clear();
    out_outcomes.reserve(states.len());
    for (s, a) in states.iter_mut().zip(actions.iter()) {
        out_outcomes.push(dynamics.step(s, a));
    }
}
