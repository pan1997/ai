//! Impartial match referee and ground-truth environment for the game of Hex.

use crate::game::{HexPlayer, HexState};
use mcts_traits::{StepOutcome, TurnBasedWorld, World};

/// Ground-truth referee and external match environment for Hex on an $N \times N$ board.
///
/// Implements [`World`] for 2-player match arbitration with path-connectivity detection,
/// while providing zero-allocation single-action step methods via [`TurnBasedWorld`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HexWorld<const N: usize = 11> {
    /// Whether the Pie (Swap) rule is active for this game.
    pub pie_rule: bool,
}

impl<const N: usize> HexWorld<N> {
    /// Creates a new `HexWorld` referee with default settings (Pie rule disabled).
    pub const fn new() -> Self {
        Self { pie_rule: false }
    }

    /// Creates a new `HexWorld` referee with the Pie rule optionally enabled.
    pub const fn with_pie_rule(pie_rule: bool) -> Self {
        Self { pie_rule }
    }

    /// Returns the board dimension $N$.
    #[inline]
    pub const fn size(&self) -> usize {
        N
    }

    /// Transitions `ws` forward by `action` for the active player without heap allocation.
    ///
    /// # Panics
    ///
    /// Panics if `action >= N * N` (unless `action == SWAP_ACTION`) or if the selected cell is already occupied.
    #[inline]
    pub fn step_action(&self, ws: &mut HexState<N>, action: usize) -> StepOutcome<[f32; 2]> {
        let idx = action;
        if idx == HexState::<N>::SWAP_ACTION {
            assert!(
                ws.pie_rule && ws.move_count == 1 && ws.current_player == HexPlayer::White,
                "Hex: SWAP_ACTION is only legal on Move 2 for White with pie rule enabled"
            );
            ws.play_swap();
            return StepOutcome::new([0.0, 0.0], false);
        }

        assert!(
            idx < N * N && ws.board[idx].is_none(),
            "Hex: cell {idx} is out of bounds or already occupied"
        );
        let player = ws.current_player;
        let won = ws.play_move(idx);

        if won {
            let reward = match player {
                HexPlayer::Black => [1.0, -1.0],
                HexPlayer::White => [-1.0, 1.0],
            };
            StepOutcome::new(reward, true)
        } else if ws.board.iter().all(|c| c.is_some()) {
            // Under optimal play Hex cannot draw, but if all cells fill without connection:
            StepOutcome::new([0.0, 0.0], true)
        } else {
            ws.current_player = player.other();
            StepOutcome::new([0.0, 0.0], false)
        }
    }

    /// Populates `out` with all legal, unoccupied cell indices on the board.
    #[inline]
    pub fn legal_actions(&self, ws: &HexState<N>, out: &mut Vec<usize>) {
        ws.legal_actions(out);
    }

    /// Returns `true` if either player has won or the board is completely filled.
    #[inline]
    pub fn is_terminal(&self, ws: &HexState<N>) -> bool {
        ws.is_terminal()
    }
}

impl<const N: usize> World for HexWorld<N> {
    type WorldState = HexState<N>;
    type Action = usize;
    type Observation = HexState<N>;

    #[inline]
    fn n_players(&self) -> usize {
        2
    }

    #[inline]
    fn initial(&self) -> Self::WorldState {
        HexState::with_pie_rule(self.pie_rule)
    }

    #[inline]
    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        ws.clone()
    }

    #[inline]
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        let active = ws.current_player.index();
        if player == active && !self.terminal(ws) {
            self.legal_actions(ws, out);
        }
    }

    #[inline]
    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        let active = ws.current_player.index();
        let outcome = self.step_action(ws, joint[active]);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    #[inline]
    fn terminal(&self, ws: &Self::WorldState) -> bool {
        self.is_terminal(ws)
    }
}

impl<const N: usize> TurnBasedWorld for HexWorld<N> {
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
