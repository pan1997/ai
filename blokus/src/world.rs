//! Ground-truth World referee implementation for Blokus.

use crate::dynamics::compute_rank_rewards;
use crate::game::{BlokusAction, BlokusState};
use mcts_traits::{StepOutcome, TurnBasedWorld, World};

/// Impartial referee and ground-truth environment for Blokus.
///
/// Models the external arbitration of a $P$-player Blokus match on a $B \times B$ board.
///
/// Standard configurations:
/// - Blokus Classic: `BlokusWorld<20, 4>` ($20 \times 20$ board, 4 players)
/// - Blokus Duo: `BlokusWorld<14, 2>` ($14 \times 14$ board, 2 players)
///
/// Implements [`World`] for joint-action tournament and match arbitration, while providing
/// inherent zero-allocation methods for single-action transitions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlokusWorld<const B: usize = 20, const P: usize = 4>;

impl<const B: usize, const P: usize> BlokusWorld<B, P> {
    /// Creates a new `BlokusWorld` referee.
    pub const fn new() -> Self {
        Self
    }

    /// Transitions `ws` forward by `action` for the active player without heap allocation.
    ///
    /// Returns the immediate reward vector $\mathbf{r} \in [-1.0, 1.0]^P$ and episode termination flag.
    ///
    /// # Panics
    ///
    /// Panics if `action` is illegal according to the rules of Blokus.
    #[inline]
    pub fn step_action(&self, ws: &mut BlokusState<B, P>, action: &BlokusAction) -> StepOutcome<[f32; P]> {
        ws.apply_action(action)
            .unwrap_or_else(|err| panic!("BlokusWorld: illegal action {action:?}: {err}"));

        if ws.is_terminal() {
            let mut scores = [0i32; P];
            for p in 0..P {
                scores[p] = ws.score(p);
            }
            let rewards = compute_rank_rewards(&scores);
            StepOutcome::new(rewards, true)
        } else {
            StepOutcome::new([0.0f32; P], false)
        }
    }

    /// Populates `out` with all legal actions for the current player in `ws`.
    #[inline]
    pub fn legal_actions(&self, ws: &BlokusState<B, P>, out: &mut Vec<BlokusAction>) {
        ws.legal_actions(out);
    }

    /// Returns `true` if the match is in a terminal state.
    #[inline]
    pub fn is_terminal(&self, ws: &BlokusState<B, P>) -> bool {
        ws.is_terminal()
    }
}

impl<const B: usize, const P: usize> World for BlokusWorld<B, P> {
    type WorldState = BlokusState<B, P>;
    type Action = BlokusAction;
    type Observation = BlokusState<B, P>;

    #[inline]
    fn n_players(&self) -> usize {
        P
    }

    #[inline]
    fn initial(&self) -> Self::WorldState {
        BlokusState::new()
    }

    #[inline]
    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        // Perfect information game: full-state observation
        ws.clone()
    }

    #[inline]
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        if player == ws.current_player as usize && !ws.is_terminal() {
            ws.legal_actions(out);
        }
    }

    #[inline]
    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool) {
        let active_player = ws.current_player as usize;
        let action = &joint[active_player];
        let outcome = self.step_action(ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    #[inline]
    fn terminal(&self, ws: &Self::WorldState) -> bool {
        self.is_terminal(ws)
    }
}

impl<const B: usize, const P: usize> TurnBasedWorld for BlokusWorld<B, P> {
    type StepReward = [f32; P];

    #[inline]
    fn current_player(&self, ws: &Self::WorldState) -> usize {
        ws.current_player as usize
    }

    #[inline]
    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> StepOutcome<Self::StepReward> {
        self.step_action(ws, action)
    }
}

/// Standard 4-player Blokus Classic referee.
pub type BlokusClassicWorld = BlokusWorld<20, 4>;

/// Standard 2-player Blokus Duo referee.
pub type BlokusDuoWorld = BlokusWorld<14, 2>;

