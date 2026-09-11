//! Dynamics and World referee implementations for Connect 4.

use crate::game::{Connect4State, Player};
use mcts_traits::{AgentDynamics, BatchedAgentDynamics, StepOutcome, World};

/// Parametric Connect 4 game dynamics with $R$ rows and $C$ columns.
///
/// Standard Connect 4 uses $R=6, C=7$.
///
/// Implements:
/// - [`AgentDynamics`]: Single-player planning transition steps.
/// - [`BatchedAgentDynamics`]: Vectorized batch steps.
/// - [`World`]: 2-player match referee with simultaneous joint-action steps.
#[derive(Debug, Clone, Copy, Default)]
pub struct Connect4Dynamics<const R: usize = 6, const C: usize = 7>;

impl<const R: usize, const C: usize> AgentDynamics for Connect4Dynamics<R, C> {
    type State = Connect4State<R, C>;
    type Action = usize;
    type Reward = [f32; 2];

    fn initial(&self) -> Self::State {
        Connect4State::new()
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        s.legal_actions(out);
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        let col = *action;
        let current_player = s.current_player;
        let placed_row = s
            .drop_piece(col)
            .unwrap_or_else(|err| panic!("Connect4: invalid action {col}: {err}"));

        let is_win = s.check_win_at(placed_row, col, current_player);
        if is_win {
            let reward = match current_player {
                Player::Red => [1.0, -1.0],
                Player::Yellow => [-1.0, 1.0],
            };
            StepOutcome::new(reward, true)
        } else if s.is_board_full() {
            StepOutcome::new([0.0, 0.0], true)
        } else {
            s.current_player = current_player.other();
            StepOutcome::new([0.0, 0.0], false)
        }
    }

    fn current_agent(&self, s: &Self::State) -> mcts_traits::AgentId {
        mcts_traits::AgentId(s.current_player.index() as u32)
    }
}

impl<const R: usize, const C: usize> BatchedAgentDynamics for Connect4Dynamics<R, C> {
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_outcomes);
    }
}

impl<const R: usize, const C: usize> World for Connect4Dynamics<R, C> {
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
        if player == active_player {
            AgentDynamics::actions(self, ws, out);
        }
    }

    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool) {
        let active_player = ws.current_player.index();
        let action = &joint[active_player];
        let outcome = AgentDynamics::step(self, ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.is_board_full() || ws.has_won(Player::Red) || ws.has_won(Player::Yellow)
    }
}
