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
    type WorldState;
    type Action: Eq + Debug;
    type Observation;

    /// Number of players/agents participating in the environment.
    fn n_players(&self) -> usize;

    /// Returns the initial world state.
    fn initial(&self) -> Self::WorldState;

    /// Generates an observation for `player` from the world state (full-state or delta).
    fn observe(&self, ws: &Self::WorldState, player: usize) -> Self::Observation;

    /// Legal actions available for `player` in the current world state.
    /// Inactive players in turn-based games return an empty list or a single Noop action.
    fn actions(&self, ws: &Self::WorldState, player: usize) -> Vec<Self::Action>;

    /// Executes joint actions from all players.
    /// Returns (next_world_state, per_player_rewards, is_terminal).
    fn step(
        &self,
        ws: Self::WorldState,
        joint: &[Self::Action],
    ) -> (Self::WorldState, Vec<f32>, bool);

    /// Checks if the world state is terminal.
    fn terminal(&self, ws: &Self::WorldState) -> bool;
}

