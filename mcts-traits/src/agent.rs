/// Strongly typed identifier for an agent or player in an environment.
///
/// Wraps a 32-bit integer (`u32`), allowing compact representation in search trees,
/// array indexing via [`AgentId::as_usize`], and interop with numeric representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AgentId(pub u32);

impl AgentId {
    /// Creates a new `AgentId` from a 32-bit unsigned integer.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Converts the agent ID to a `usize` for array indexing.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns the raw `u32` value of the agent ID.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for AgentId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for AgentId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<AgentId> for usize {
    #[inline]
    fn from(agent: AgentId) -> Self {
        agent.as_usize()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Agent({})", self.0)
    }
}

/// General agent interface capable of choosing actions in an environment state.
///
/// Models an active decision maker in real games, self-play arenas, or tournament drivers.
/// The generic parameter `Delta` represents transition metadata or chance outcomes (e.g. tile spawns),
/// defaulting to `()` for deterministic turn-based environments.
pub trait Agent<State, Action, Delta = ()> {
    /// Human-readable name or label of the agent.
    fn name(&self) -> &str;

    /// Selects an action given the current environment state.
    fn select_action(&mut self, state: &State) -> Action;

    /// Selects an action given the current environment state and the sequence
    /// of `(Action, Delta)` transitions that took place in the environment
    /// since this agent's previous decision.
    ///
    /// The default implementation forwards directly to [`Agent::select_action`].
    /// Stateful agents (e.g. MCTS agents with persistent search trees) can override
    /// this method to advance their tree root and call `promote_subtree`.
    #[inline]
    fn select_action_with_history(
        &mut self,
        state: &State,
        _history: &[(Action, Delta)],
    ) -> Action {
        self.select_action(state)
    }

    /// Resets persistent search trees, transposition tables, or history at the start of a match.
    ///
    /// The default implementation is a no-op.
    #[inline]
    fn reset(&mut self) {}
}

impl<State, Action, Delta, A: Agent<State, Action, Delta> + ?Sized> Agent<State, Action, Delta>
    for Box<A>
{
    #[inline]
    fn name(&self) -> &str {
        (**self).name()
    }

    #[inline]
    fn select_action(&mut self, state: &State) -> Action {
        (**self).select_action(state)
    }

    #[inline]
    fn select_action_with_history(&mut self, state: &State, history: &[(Action, Delta)]) -> Action {
        (**self).select_action_with_history(state, history)
    }

    #[inline]
    fn reset(&mut self) {
        (**self).reset();
    }
}

impl<State, Action, Delta, A: Agent<State, Action, Delta> + ?Sized> Agent<State, Action, Delta>
    for &mut A
{
    #[inline]
    fn name(&self) -> &str {
        (**self).name()
    }

    #[inline]
    fn select_action(&mut self, state: &State) -> Action {
        (**self).select_action(state)
    }

    #[inline]
    fn select_action_with_history(&mut self, state: &State, history: &[(Action, Delta)]) -> Action {
        (**self).select_action_with_history(state, history)
    }

    #[inline]
    fn reset(&mut self) {
        (**self).reset();
    }
}
