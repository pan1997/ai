use std::fmt::Debug;

/// Outcome of stepping an environment state in-place with an action.
///
/// Encapsulates the immediate reward signal received, the transition delta (metadata/observation),
/// and the episode termination flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepOutcome<R, D = ()> {
    /// Immediate transition reward received (e.g., scalar return or per-agent reward vector).
    pub reward: R,
    /// Transition metadata or observation (e.g. opponent action, chance outcome, or ()).
    pub delta: D,
    /// Indicates whether the subsequent state is a terminal game/episode state.
    pub terminated: bool,
}

impl<R> StepOutcome<R, ()> {
    /// Constructs a new `StepOutcome` with unit delta `()`.
    #[inline]
    pub fn new(reward: R, terminated: bool) -> Self {
        Self {
            reward,
            delta: (),
            terminated,
        }
    }
}

impl<R, D> StepOutcome<R, D> {
    /// Constructs a new `StepOutcome` with an explicit transition `delta`.
    #[inline]
    pub fn with_delta(reward: R, delta: D, terminated: bool) -> Self {
        Self {
            reward,
            delta,
            terminated,
        }
    }
}

/// Result of stepping an environment state with an action, holding ownership of the next state.
///
/// Primarily used for transition logging, experience replay buffers, or non-destructive simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition<S, R, D = ()> {
    /// The resulting state after taking the action.
    pub next_state: S,
    /// Immediate transition reward received (e.g., scalar return or per-agent reward vector).
    pub reward: R,
    /// Transition metadata or observation (e.g. opponent action, chance outcome, or ()).
    pub delta: D,
    /// Indicates whether the next state is a terminal game/episode state.
    pub terminated: bool,
}

impl<S, R> Transition<S, R, ()> {
    /// Constructs a new `Transition` with unit delta `()`.
    #[inline]
    pub fn new(next_state: S, reward: R, terminated: bool) -> Self {
        Self {
            next_state,
            reward,
            delta: (),
            terminated,
        }
    }
}

impl<S, R, D> Transition<S, R, D> {
    /// Constructs a new `Transition` with an explicit transition `delta`.
    #[inline]
    pub fn with_delta(next_state: S, reward: R, delta: D, terminated: bool) -> Self {
        Self {
            next_state,
            reward,
            delta,
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
    /// Transition delta or observation metadata (e.g. opponent action, chance outcome, or ()).
    type StepDelta: Eq + Clone + Debug;

    /// Returns the initial or root state for planning.
    fn initial(&self) -> Self::State;

    /// Generates legal actions available in state `s` into `out`.
    ///
    /// Clears or appends to `out` to guarantee zero heap allocations during search tree expansion.
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>);

    /// Transitions state `s` forward in-place given `action`.
    ///
    /// Modifies `s` directly without heap allocation or intermediate cloning, returning the
    /// immediate reward, transition delta, and termination status.
    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta>;

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
        out_outcomes: &mut Vec<StepOutcome<Self::Reward, Self::StepDelta>>,
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
    out_outcomes: &mut Vec<StepOutcome<D::Reward, D::StepDelta>>,
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
    type StepDelta = ();

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
    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward, ()> {
        let outcome = self.world.step_action(s, action);
        StepOutcome::new(outcome.reward, outcome.terminated)
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
        out_outcomes: &mut Vec<StepOutcome<Self::Reward, ()>>,
    ) {
        default_step_batch(self, states, actions, out_outcomes);
    }
}

/// Trait governing opponent behavior given an environment state where an opponent is to act.
///
/// Implementations can represent heuristic policies (e.g. immediate win/block tactical policies),
/// uniform random selection, pre-trained policy networks, or default responses.
pub trait OpponentPolicy<State, Action>: Send + Sync {
    /// Selects an action for the active opponent in `state`.
    fn select_action(&self, state: &State) -> Action;

    /// Populates prior probabilities for candidate actions when expanding an afterstate.
    ///
    /// The default implementation assigns uniform probabilities $P(a) = \frac{1}{|\mathcal{A}|}$.
    fn action_priors(&self, _state: &State, legal_actions: &[Action], out_priors: &mut Vec<f32>) {
        out_priors.clear();
        let n = legal_actions.len();
        if n > 0 {
            out_priors.resize(n, 1.0 / n as f32);
        }
    }
}

/// Universal round-based macro-dynamics adapter wrapping a [`TurnBasedWorld`](crate::world::TurnBasedWorld) and an opponent policy `P`.
///
/// In multi-agent games, planning across full game rounds rather than individual half-moves (plies) provides
/// multiple advantages:
/// - **Depth Expansion**: Lookahead depth $d$ corresponds to $d$ complete game rounds.
/// - **Evaluator Single-Perspective**: Value and policy models only ever need to evaluate states from
///   the perspective of `primary_player`, eliminating evaluator dilution.
///
/// Each call to `step` executes the primary agent's action, then absorbs opponent replies via `P::select_action`
/// until `current_player == primary_player` or the episode terminates, emitting the opponent's action in `StepDelta`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoundBasedDynamics<W, P> {
    /// The underlying ground-truth world referee.
    pub world: W,
    /// Policy governing opponent actions during afterstate transitions.
    pub opponent_policy: P,
    /// Index of the primary planning player ($0 \le \text{primary\_player} < \text{n\_players}$).
    pub primary_player: usize,
}

impl<W, P> RoundBasedDynamics<W, P> {
    /// Creates a new `RoundBasedDynamics` adapter.
    #[inline]
    pub const fn new(world: W, opponent_policy: P, primary_player: usize) -> Self {
        Self {
            world,
            opponent_policy,
            primary_player,
        }
    }

    /// Returns a reference to the wrapped world referee.
    #[inline]
    pub const fn world(&self) -> &W {
        &self.world
    }

    /// Returns a reference to the opponent policy.
    #[inline]
    pub const fn opponent_policy(&self) -> &P {
        &self.opponent_policy
    }

    /// Returns the primary player index.
    #[inline]
    pub const fn primary_player(&self) -> usize {
        self.primary_player
    }
}

impl<W, P> AgentDynamics for RoundBasedDynamics<W, P>
where
    W: crate::world::TurnBasedWorld,
    W::Action: Clone,
    P: OpponentPolicy<W::WorldState, W::Action>,
{
    type State = W::WorldState;
    type Action = W::Action;
    type Reward = W::StepReward;
    type StepDelta = Option<W::Action>;

    #[inline]
    fn initial(&self) -> Self::State {
        self.world.initial()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        self.world.actions(s, self.primary_player, out);
    }

    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        // 1. Apply primary agent's action
        let primary_outcome = self.world.step_action(s, action);
        if primary_outcome.terminated {
            return StepOutcome::with_delta(primary_outcome.reward, None, true);
        }

        // 2. Loop through all subsequent opponent turns until returning to primary player or terminal
        let mut last_opp_action = None;
        while !self.world.terminal(s) && self.world.current_player(s) != self.primary_player {
            let opp_action = self.opponent_policy.select_action(s);
            let outcome = self.world.step_action(s, &opp_action);
            last_opp_action = Some(opp_action);
            if outcome.terminated {
                return StepOutcome::with_delta(outcome.reward, last_opp_action, true);
            }
        }

        StepOutcome::with_delta(
            primary_outcome.reward,
            last_opp_action,
            self.world.terminal(s),
        )
    }

    #[inline]
    fn current_agent(&self, _s: &Self::State) -> crate::AgentId {
        crate::AgentId(self.primary_player as u32)
    }
}

impl<W, P> BatchedAgentDynamics for RoundBasedDynamics<W, P>
where
    W: crate::world::TurnBasedWorld,
    W::Action: Clone,
    P: OpponentPolicy<W::WorldState, W::Action>,
{
    #[inline]
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward, Self::StepDelta>>,
    ) {
        default_step_batch(self, states, actions, out_outcomes);
    }
}
