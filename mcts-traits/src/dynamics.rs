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

    /// Returns the active agent ID whose turn it is to act in state `s`.
    ///
    /// Defaults to `AgentId(0)` for single-agent environments.
    #[inline]
    fn current_agent(&self, _s: &Self::State) -> crate::AgentId {
        crate::AgentId(0)
    }
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

/// Universal zero-cost planning dynamics adapter for any [`TurnBasedWorld`](crate::world::TurnBasedWorld).
///
/// Automatically bridges an impartial multi-player referee ([`World`](crate::world::World)) into an
/// agent-centric planning transition model ([`AgentDynamics`]) with zero heap allocation.
///
/// ### Architecture & Invariants:
/// - **Zero Heap Allocation**: Traversal through `AgentDynamics::step` passes through to
///   [`crate::world::TurnBasedWorld::step_action`], which updates the state in-place and returns on the stack.
/// - **Active Agent Tracking**: `AgentDynamics::current_agent` queries [`crate::world::TurnBasedWorld::current_player`],
///   ensuring that MCTS selection algorithms (e.g. [`MultiAgentPuctSelection`](https://docs.rs/mcts-engine))
///   maximize the return component corresponding to the active player.
/// - **Refactoring Simplification**: Replaces redundant handwritten wrapper types (e.g. `Connect4Dynamics`,
///   `BlokusDynamics`, `HexDynamics`) with a single generic adapter.
///
/// ### Example
/// ```rust,ignore
/// use mcts_traits::TurnBasedDynamics;
/// use connect4::Connect4World;
///
/// let env = TurnBasedDynamics::new(Connect4World::<6, 7>::new());
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TurnBasedDynamics<W> {
    /// The underlying ground-truth world referee.
    pub world: W,
}

impl<W> TurnBasedDynamics<W> {
    /// Creates a new `TurnBasedDynamics` adapter wrapping `world`.
    #[inline]
    pub const fn new(world: W) -> Self {
        Self { world }
    }

    /// Returns a reference to the wrapped world referee.
    #[inline]
    pub const fn world(&self) -> &W {
        &self.world
    }
}

impl<W: crate::world::TurnBasedWorld> AgentDynamics for TurnBasedDynamics<W> {
    type State = W::WorldState;
    type Action = W::Action;
    type Reward = W::StepReward;

    #[inline]
    fn initial(&self) -> Self::State {
        self.world.initial()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        let player = self.world.current_player(s);
        self.world.actions(s, player, out);
    }

    #[inline]
    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        self.world.step_action(s, action)
    }

    #[inline]
    fn current_agent(&self, s: &Self::State) -> crate::AgentId {
        crate::AgentId(self.world.current_player(s) as u32)
    }
}

impl<W: crate::world::TurnBasedWorld> BatchedAgentDynamics for TurnBasedDynamics<W> {
    #[inline]
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
    ) {
        default_step_batch(self, states, actions, out_outcomes);
    }
}
