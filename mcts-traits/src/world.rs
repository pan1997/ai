use std::fmt::Debug;

/// External world and referee interface for real matches, self-play, and tournament drivers.
///
/// ### Design Distinction:
/// - `World` is the impartial ground truth (manages full hidden card decks, refereeing, simultaneous moves).
/// - `AgentDynamics` is the agent's internal planning model (single-agent hypothetical reasoning).
///
/// ### Observation Granularity:
/// Depending on the environment implementation, `Observation` can be:
/// - **Full-State Observation**: Complete snapshot of public state + player's private hand
///   (common in card games like Poker or board games).
/// - **Delta Observation**: Incremental event log since the player's last decision
///   (e.g. "Opponent raised by 50", common in streaming or continuous games).
pub trait World {
    /// The complete ground-truth representation of the game/environment.
    type WorldState;
    /// Action representation for individual player moves.
    type Action: Eq + Debug;
    /// Player-specific observation type (e.g. filtered view or event delta).
    type Observation;

    /// Number of players/agents participating in the environment.
    fn n_players(&self) -> usize;

    /// Returns the initial ground-truth world state.
    fn initial(&self) -> Self::WorldState;

    /// Generates a player-specific observation from `ws` (full-state snapshot or delta).
    fn observe(&self, ws: &Self::WorldState, player: usize) -> Self::Observation;

    /// Populates `out` with legal actions available for `player` in the current world state.
    ///
    /// Clears or appends to `out` to guarantee zero heap allocations during arbitration loops.
    /// Inactive players in turn-based games return an empty list or a single Noop action.
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>);

    /// Executes joint actions from all players simultaneously in-place on `ws`.
    ///
    /// Modifies `ws` directly and returns `(per_player_rewards, is_terminal)`.
    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool);

    /// Checks if the world state is in a terminal condition.
    fn terminal(&self, ws: &Self::WorldState) -> bool;
}

/// Extension trait for sequential turn-based environments with zero-allocation stack transitions.
///
/// Many classical board games (e.g. Connect 4, Hex, Blokus, Chess, Go) are sequential, perfect-information,
/// turn-based games where players take turns choosing single moves.
///
/// While [`World`] requires a simultaneous joint-action vector `joint: &[Self::Action]` (returning a heap-allocated `Vec<f32>`),
/// high-performance MCTS search requires stepping single moves in-place with **zero heap allocations**.
///
/// Implementors declare whose turn it is ([`TurnBasedWorld::current_player`]) and provide
/// a stack-allocated single-action transition ([`TurnBasedWorld::step_action`]).
///
/// Implementing this trait automatically enables universal conversion into an
/// [`AgentDynamics`](crate::AgentDynamics) via [`TurnBasedDynamics`](crate::dynamics::TurnBasedDynamics).
///
/// ### Example
/// ```rust,ignore
/// use mcts_traits::{TurnBasedWorld, TurnBasedDynamics};
///
/// let world = Connect4World::new();
/// let dynamics = TurnBasedDynamics::new(world);
/// ```
pub trait TurnBasedWorld: World {
    /// Return representation for zero-allocation step outcomes (typically a fixed-size array `[f32; N]`).
    type StepReward;

    /// Returns the index of the player whose turn it is to act in `ws` ($0 \le \text{player} < \text{n\_players}$).
    fn current_player(&self, ws: &Self::WorldState) -> usize;

    /// Transitions `ws` forward by `action` for the active player in-place without heap allocation.
    ///
    /// # Arguments
    /// * `ws` - Mutable reference to the current world state to advance.
    /// * `action` - The action chosen by `self.current_player(ws)`.
    ///
    /// # Returns
    /// Returns a [`StepOutcome`](crate::dynamics::StepOutcome) containing the immediate reward vector
    /// $\mathbf{r} \in \mathbb{R}^N$ and whether the episode has reached a terminal state.
    ///
    /// # Panics
    /// Panics if `action` is not legal in `ws` for the active player.
    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> crate::dynamics::StepOutcome<Self::StepReward>;
}
