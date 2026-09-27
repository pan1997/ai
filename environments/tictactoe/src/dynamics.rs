//! Agent dynamics implementation for TicTacToe MCTS planning.

use crate::game::TicTacToeState;
use mcts_traits::{AgentDynamics, AgentId, BatchedAgentDynamics, StepOutcome};

/// Planning dynamics model for TicTacToe.
#[derive(Debug, Clone, Copy, Default)]
pub struct TicTacToeDynamics;

impl AgentDynamics for TicTacToeDynamics {
    type State = TicTacToeState;
    type Action = usize;
    type Reward = [f32; 2];
    type StepDelta = ();

    #[inline]
    fn initial(&self) -> Self::State {
        TicTacToeState::new()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        s.legal_actions(out);
    }

    #[inline]
    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        s.apply_action(*action);
        let terminated = s.is_terminal();
        let reward = s.outcome().unwrap_or([0.0, 0.0]);
        StepOutcome::new(reward, terminated)
    }

    #[inline]
    fn current_agent(&self, s: &Self::State) -> AgentId {
        AgentId(s.current_player.index() as u32)
    }
}

impl BatchedAgentDynamics for TicTacToeDynamics {
    #[inline]
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward, Self::StepDelta>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_outcomes);
    }
}
