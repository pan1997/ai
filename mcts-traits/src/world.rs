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
    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool);

    /// Checks if the world state is in a terminal condition.
    fn terminal(&self, ws: &Self::WorldState) -> bool;
}

