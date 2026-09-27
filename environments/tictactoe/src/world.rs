//! Ground-truth World referee implementation for TicTacToe.

use crate::game::TicTacToeState;
use mcts_traits::{StepOutcome, TurnBasedWorld, World};

/// Impartial referee and ground-truth environment for 2-player TicTacToe.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TicTacToeWorld;

impl TicTacToeWorld {
    /// Creates a new `TicTacToeWorld` referee.
    pub const fn new() -> Self {
        Self
    }

    /// Transitions `ws` forward by `action` for the active player.
    #[inline]
    pub fn step_action(&self, ws: &mut TicTacToeState, action: usize) -> StepOutcome<[f32; 2]> {
        ws.apply_action(action);
        let terminated = ws.is_terminal();
        let reward = ws.outcome().unwrap_or([0.0, 0.0]);
        StepOutcome::new(reward, terminated)
    }

    /// Populates `out` with all legal action indices in `ws`.
    #[inline]
    pub fn legal_actions(&self, ws: &TicTacToeState, out: &mut Vec<usize>) {
        ws.legal_actions(out);
    }

    /// Returns `true` if the game is over.
    #[inline]
    pub fn is_terminal(&self, ws: &TicTacToeState) -> bool {
        ws.is_terminal()
    }
}

impl World for TicTacToeWorld {
    type WorldState = TicTacToeState;
    type Action = usize;
    type Observation = TicTacToeState;

    fn n_players(&self) -> usize {
        2
    }

    fn initial(&self) -> Self::WorldState {
        TicTacToeState::new()
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
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

impl TurnBasedWorld for TicTacToeWorld {
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
