//! Dynamics and World referee implementations for Blokus.

use crate::game::{BlokusAction, BlokusState};
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
/// Standard configurations:
/// - Blokus Classic: `BlokusDynamics<20, 4>` ($20 \times 20$ board, 4 players)
/// - Blokus Duo: `BlokusDynamics<14, 2>` ($14 \times 14$ board, 2 players)
///
/// Implements:
/// - [`AgentDynamics`]: Single-agent hypothetical transition steps for MCTS planning.
/// - [`BatchedAgentDynamics`]: Vectorized batch transitions.
/// - [`World`]: Impartial ground-truth referee managing multi-player matches and tournaments.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlokusDynamics<const B: usize = 20, const P: usize = 4>;

impl<const B: usize, const P: usize> AgentDynamics for BlokusDynamics<B, P> {
    type State = BlokusState<B, P>;
    type Action = BlokusAction;
    type Reward = [f32; P];

    fn initial(&self) -> Self::State {
        BlokusState::new()
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        s.legal_actions(out);
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        s.apply_action(action)
            .unwrap_or_else(|err| panic!("BlokusDynamics: illegal action {action:?}: {err}"));

        if s.is_terminal() {
            let mut scores = [0i32; P];
            for p in 0..P {
                scores[p] = s.score(p);
            }
            let rewards = compute_rank_rewards(&scores);
            StepOutcome::new(rewards, true)
        } else {
            StepOutcome::new([0.0f32; P], false)
        }
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

    fn n_players(&self) -> usize {
        P
    }

    fn initial(&self) -> Self::WorldState {
        BlokusState::new()
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        // Perfect information game: full-state observation
        ws.clone()
    }

    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        if player == ws.current_player as usize && !ws.is_terminal() {
            ws.legal_actions(out);
        }
    }

    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool) {
        let active_player = ws.current_player as usize;
        let action = &joint[active_player];
        let outcome = AgentDynamics::step(self, ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.is_terminal()
    }
}

/// Standard 4-player Blokus Classic dynamics.
pub type BlokusClassicDynamics = BlokusDynamics<20, 4>;

/// Standard 2-player Blokus Duo dynamics.
pub type BlokusDuoDynamics = BlokusDynamics<14, 2>;

