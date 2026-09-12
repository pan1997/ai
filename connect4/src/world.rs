//! Ground-truth World referee implementation for Connect 4.

use crate::game::{Connect4State, Player};
use mcts_traits::{StepOutcome, World};

/// Impartial referee and ground-truth environment for Connect 4.
///
/// Models the external arbitration of a 2-player Connect 4 match with $R$ rows and $C$ columns.
/// Standard Connect 4 uses $R=6, C=7$.
///
/// Implements [`World`] for joint-action tournament and match arbitration, while providing
/// inherent zero-allocation methods for single-action transitions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Connect4World<const R: usize = 6, const C: usize = 7>;

impl<const R: usize, const C: usize> Connect4World<R, C> {
    /// Creates a new `Connect4World` referee.
    pub const fn new() -> Self {
        Self
    }

    /// Transitions `ws` forward by `action` for the active player without heap allocation.
    ///
    /// Returns the immediate reward vector `[R_red, R_yellow]` and episode termination flag.
    ///
    /// # Panics
    ///
    /// Panics if `action >= C` or if the selected column is full.
    #[inline]
    pub fn step_action(
        &self,
        ws: &mut Connect4State<R, C>,
        action: usize,
    ) -> StepOutcome<[f32; 2]> {
        let col = action;
        let current_player = ws.current_player;
        let placed_row = ws
            .drop_piece(col)
            .unwrap_or_else(|err| panic!("Connect4: invalid action {col}: {err}"));

        let is_win = ws.check_win_at(placed_row, col, current_player);
        if is_win {
            let reward = match current_player {
                Player::Red => [1.0, -1.0],
                Player::Yellow => [-1.0, 1.0],
            };
            StepOutcome::new(reward, true)
        } else if ws.is_board_full() {
            StepOutcome::new([0.0, 0.0], true)
        } else {
            ws.current_player = current_player.other();
            StepOutcome::new([0.0, 0.0], false)
        }
    }

    /// Populates `out` with all legal columns that are not full in `ws`.
    #[inline]
    pub fn legal_actions(&self, ws: &Connect4State<R, C>, out: &mut Vec<usize>) {
        ws.legal_actions(out);
    }

    /// Returns `true` if the board is full or either player has won.
    #[inline]
    pub fn is_terminal(&self, ws: &Connect4State<R, C>) -> bool {
        ws.is_board_full() || ws.has_won(Player::Red) || ws.has_won(Player::Yellow)
    }
}

impl<const R: usize, const C: usize> World for Connect4World<R, C> {
    type WorldState = Connect4State<R, C>;
    type Action = usize;
    type Observation = Connect4State<R, C>;

    fn n_players(&self) -> usize {
        2
    }

    fn initial(&self) -> Self::WorldState {
        Connect4State::new()
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        // Perfect information game: full state observation
        ws.clone()
    }

    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        let active_player = ws.current_player.index();
        if player == active_player && !self.terminal(ws) {
            self.legal_actions(ws, out);
        }
    }

    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        let active_player = ws.current_player.index();
        let action = joint[active_player];
        let outcome = self.step_action(ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        self.is_terminal(ws)
    }
}

impl<const R: usize, const C: usize> mcts_traits::TurnBasedWorld for Connect4World<R, C> {
    type StepReward = [f32; 2];

    #[inline]
    fn current_player(&self, ws: &Self::WorldState) -> usize {
        ws.current_player.index()
    }

    #[inline]
    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> StepOutcome<Self::StepReward> {
        self.step_action(ws, *action)
    }
}
