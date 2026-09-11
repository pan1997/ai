//! Planning dynamics implementations for Blokus.
//!
//! Provides [`BlokusDynamics`], a lightweight planning adapter around [`BlokusWorld`].

use crate::game::{BlokusAction, BlokusState};
use crate::world::BlokusWorld;
use mcts_traits::{AgentDynamics, AgentId, BatchedAgentDynamics, StepOutcome, World};

/// Computes normalized fractional rank rewards for $P$ players given their final scores.
///
/// Returns values in $[-1.0, 1.0]$ summing to $0.0$, gracefully handling multi-way ties.
pub fn compute_rank_rewards<const P: usize>(scores: &[i32; P]) -> [f32; P] {
    let mut rewards = [0.0f32; P];
    if P <= 1 {
        return rewards;
    }

    let denom = (P - 1) as f32;
    for i in 0..P {
        let mut strictly_lower = 0;
        let mut strictly_higher = 0;
        for j in 0..P {
            if i != j {
                if scores[i] > scores[j] {
                    strictly_lower += 1;
                } else if scores[i] < scores[j] {
                    strictly_higher += 1;
                }
            }
        }
        rewards[i] = (strictly_lower as f32 - strictly_higher as f32) / denom;
    }
    rewards
}

/// Parametric Blokus game dynamics with board dimension $B \times B$ and $P$ players.
///
/// Wraps the ground-truth [`BlokusWorld`] referee, exposing an [`AgentDynamics`] interface
/// where each step advances 1 turn according to the official rules of Blokus.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlokusDynamics<const B: usize = 20, const P: usize = 4>;

impl<const B: usize, const P: usize> BlokusDynamics<B, P> {
    /// Creates a new `BlokusDynamics` instance.
    pub const fn new() -> Self {
        Self
    }

    /// Returns the underlying [`BlokusWorld`] referee.
    pub const fn world(&self) -> BlokusWorld<B, P> {
        BlokusWorld
    }
}

impl<const B: usize, const P: usize> AgentDynamics for BlokusDynamics<B, P> {
    type State = BlokusState<B, P>;
    type Action = BlokusAction;
    type Reward = [f32; P];

    #[inline]
    fn initial(&self) -> Self::State {
        BlokusWorld::<B, P>.initial()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        BlokusWorld::<B, P>.legal_actions(s, out);
    }

    #[inline]
    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        BlokusWorld::<B, P>.step_action(s, action)
    }

    #[inline]
    fn current_agent(&self, s: &Self::State) -> AgentId {
        AgentId(s.current_player as u32)
    }
}

impl<const B: usize, const P: usize> BatchedAgentDynamics for BlokusDynamics<B, P> {
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_outcomes);
    }
}

impl<const B: usize, const P: usize> World for BlokusDynamics<B, P> {
    type WorldState = BlokusState<B, P>;
    type Action = BlokusAction;
    type Observation = BlokusState<B, P>;

    #[inline]
    fn n_players(&self) -> usize {
        BlokusWorld::<B, P>.n_players()
    }

    #[inline]
    fn initial(&self) -> Self::WorldState {
        BlokusWorld::<B, P>.initial()
    }

    #[inline]
    fn observe(&self, ws: &Self::WorldState, player: usize) -> Self::Observation {
        BlokusWorld::<B, P>.observe(ws, player)
    }

    #[inline]
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        BlokusWorld::<B, P>.actions(ws, player, out);
    }

    #[inline]
    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool) {
        BlokusWorld::<B, P>.step(ws, joint)
    }

    #[inline]
    fn terminal(&self, ws: &Self::WorldState) -> bool {
        BlokusWorld::<B, P>.terminal(ws)
    }
}

/// Standard 4-player Blokus Classic dynamics.
pub type BlokusClassicDynamics = BlokusDynamics<20, 4>;

/// Standard 2-player Blokus Duo dynamics.
pub type BlokusDuoDynamics = BlokusDynamics<14, 2>;
