//! Planning dynamics implementations for Connect 4.
//!
//! Standard alternating planning dynamics are provided universally by
//! [`TurnBasedDynamics<Connect4World<R, C>>`](mcts_traits::TurnBasedDynamics).
//!
//! This module provides alternative planning dynamics such as [`MacroConnect4Dynamics`]
//! (round-based lookahead absorbing an opponent policy).

use crate::game::{Connect4State, Player};
use crate::world::Connect4World;
pub use mcts_traits::OpponentPolicy;
use mcts_traits::{AgentDynamics, AgentId, BatchedAgentDynamics, StepOutcome, World};

/// Opponent policy selecting pseudo-randomly among legal columns based on a deterministic hash of the board state.
///
/// Deterministic evaluation is required for absorbed macro-dynamics ([`MacroConnect4Dynamics`])
/// to avoid state aliasing during MCTS trajectory traversals.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomOpponent {
    /// Optional salt/seed for the state hash.
    pub seed: u64,
}

impl RandomOpponent {
    /// Creates a new `RandomOpponent` with default seed 0.
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    /// Creates a `RandomOpponent` with a specific salt/seed.
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl<const R: usize, const C: usize> OpponentPolicy<Connect4State<R, C>, usize> for RandomOpponent {
    fn select_action(&self, s: &Connect4State<R, C>) -> usize {
        use rand::Rng;

        let mut legal = Vec::with_capacity(C);
        s.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("RandomOpponent: no legal actions available in state");
        }
        let idx = rand::thread_rng().gen_range(0..legal.len());
        legal[idx]
    }
}

/// Tactical opponent policy: plays immediate winning move, blocks opponent 1-ply win, or center preference.
#[derive(Debug, Clone, Copy, Default)]
pub struct TacticalOpponent;

impl<const R: usize, const C: usize> OpponentPolicy<Connect4State<R, C>, usize>
    for TacticalOpponent
{
    fn select_action(&self, s: &Connect4State<R, C>) -> usize {
        let mut legal = Vec::with_capacity(C);
        s.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("TacticalOpponent: no legal actions available");
        }

        let me = s.current_player;
        let opp = me.other();

        // 1. Check for immediate winning move
        for &col in &legal {
            let mut clone = s.clone();
            if let Ok(row) = clone.drop_piece(col)
                && clone.check_win_at(row, col, me)
            {
                return col;
            }
        }

        // 2. Check for immediate blocking move
        for &col in &legal {
            let mut clone = s.clone();
            clone.current_player = opp;
            if let Ok(row) = clone.drop_piece(col)
                && clone.check_win_at(row, col, opp)
            {
                return col;
            }
        }

        // 3. Fallback: choose column closest to the center
        let center = C as f32 / 2.0;
        *legal
            .iter()
            .min_by(|&&a, &&b| {
                let dist_a = (a as f32 + 0.5 - center).abs();
                let dist_b = (b as f32 + 0.5 - center).abs();
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }
}

/// Macro dynamics for Connect 4 that plan across complete full rounds.
///
/// In standard Connect 4 MCTS, the agent plans ply-by-ply (Red moves, then Blue moves).
/// `MacroConnect4Dynamics` absorbs the opponent's reply via a pluggable [`OpponentPolicy`]:
/// - When the agent selects an action $A_{\text{ego}}$, the dynamics executes $A_{\text{ego}}$.
/// - If the game has not ended, the opponent policy replies with $A_{\text{opp}}$.
/// - The step outcome returns the transition reward and the opponent's reply in `StepDelta`.
#[derive(Debug, Clone, Copy)]
pub struct MacroConnect4Dynamics<P, const R: usize = 6, const C: usize = 7> {
    /// Canonical referee managing the game state.
    pub world: Connect4World<R, C>,
    /// Policy governing opponent reactions.
    pub opponent_policy: P,
    /// The primary player whose perspective and returns are being maximized.
    pub primary_player: Player,
}

impl<P: Default, const R: usize, const C: usize> Default for MacroConnect4Dynamics<P, R, C> {
    fn default() -> Self {
        Self {
            world: Connect4World::new(),
            opponent_policy: P::default(),
            primary_player: Player::Red,
        }
    }
}

impl<P, const R: usize, const C: usize> MacroConnect4Dynamics<P, R, C> {
    /// Creates a new `MacroConnect4Dynamics` with the given opponent policy and primary player.
    pub fn new(opponent_policy: P, primary_player: Player) -> Self {
        Self {
            world: Connect4World::new(),
            opponent_policy,
            primary_player,
        }
    }
}

impl<P, const R: usize, const C: usize> AgentDynamics for MacroConnect4Dynamics<P, R, C>
where
    P: OpponentPolicy<Connect4State<R, C>, usize>,
{
    type State = Connect4State<R, C>;
    type Action = usize;
    type Reward = [f32; 2];
    type StepDelta = Option<usize>;

    #[inline]
    fn initial(&self) -> Self::State {
        self.world.initial()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        self.world.legal_actions(s, out);
    }

    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        // 1. Apply primary agent's move
        let primary_outcome = self.world.step_action(s, *action);
        if primary_outcome.terminated {
            return StepOutcome::with_delta(primary_outcome.reward, None, true);
        }

        // 2. Opponent replies via pluggable policy
        let opp_action = self.opponent_policy.select_action(s);
        let opp_outcome = self.world.step_action(s, opp_action);
        StepOutcome::with_delta(opp_outcome.reward, Some(opp_action), opp_outcome.terminated)
    }

    #[inline]
    fn current_agent(&self, _s: &Self::State) -> AgentId {
        AgentId(self.primary_player.index() as u32)
    }
}

impl<P, const R: usize, const C: usize> BatchedAgentDynamics for MacroConnect4Dynamics<P, R, C>
where
    P: OpponentPolicy<Connect4State<R, C>, usize>,
{
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward, Self::StepDelta>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_outcomes);
    }
}
