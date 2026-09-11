//! Planning dynamics implementations for Connect 4.
//!
//! Provides [`Connect4Dynamics`], a lightweight planning adapter around [`Connect4World`],
//! as well as alternative dynamics such as [`MacroConnect4Dynamics`] (round-based lookahead
//! absorbing an opponent policy).

use crate::game::{Connect4State, Player};
use crate::world::Connect4World;
use mcts_traits::{AgentDynamics, AgentId, BatchedAgentDynamics, StepOutcome, World};
use rand::seq::SliceRandom;

/// Standard alternating-turn Connect 4 planning dynamics.
///
/// Wraps the ground-truth [`Connect4World`] referee, exposing an [`AgentDynamics`] interface
/// where each step advances 1 ply and toggles the active player.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Connect4Dynamics<const R: usize = 6, const C: usize = 7>;

impl<const R: usize, const C: usize> Connect4Dynamics<R, C> {
    /// Creates a new `Connect4Dynamics` instance.
    pub const fn new() -> Self {
        Self
    }

    /// Returns the underlying [`Connect4World`] referee.
    pub const fn world(&self) -> Connect4World<R, C> {
        Connect4World
    }
}

impl<const R: usize, const C: usize> AgentDynamics for Connect4Dynamics<R, C> {
    type State = Connect4State<R, C>;
    type Action = usize;
    type Reward = [f32; 2];

    #[inline]
    fn initial(&self) -> Self::State {
        Connect4World::<R, C>.initial()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        Connect4World::<R, C>.legal_actions(s, out);
    }

    #[inline]
    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        Connect4World::<R, C>.step_action(s, *action)
    }

    #[inline]
    fn current_agent(&self, s: &Self::State) -> AgentId {
        AgentId(s.current_player.index() as u32)
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

    #[inline]
    fn n_players(&self) -> usize {
        Connect4World::<R, C>.n_players()
    }

    #[inline]
    fn initial(&self) -> Self::WorldState {
        Connect4World::<R, C>.initial()
    }

    #[inline]
    fn observe(&self, ws: &Self::WorldState, player: usize) -> Self::Observation {
        Connect4World::<R, C>.observe(ws, player)
    }

    #[inline]
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        Connect4World::<R, C>.actions(ws, player, out);
    }

    #[inline]
    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        Connect4World::<R, C>.step(ws, joint)
    }

    #[inline]
    fn terminal(&self, ws: &Self::WorldState) -> bool {
        Connect4World::<R, C>.terminal(ws)
    }
}

/// Pluggable policy governing opponent responses in macro-action dynamics.
pub trait OpponentPolicy<const R: usize, const C: usize>: Send + Sync {
    /// Selects an action for the opponent given state `s` where `s.current_player` is the opponent.
    fn select_action(&self, s: &Connect4State<R, C>) -> usize;
}

/// Opponent policy selecting uniformly at random among legal columns.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomOpponent;

impl<const R: usize, const C: usize> OpponentPolicy<R, C> for RandomOpponent {
    fn select_action(&self, s: &Connect4State<R, C>) -> usize {
        let mut legal = Vec::with_capacity(C);
        s.legal_actions(&mut legal);
        let mut rng = rand::thread_rng();
        *legal.choose(&mut rng).expect("opponent has legal move")
    }
}

/// Tactical opponent policy: plays immediate winning move, blocks opponent 1-ply win, or center preference.
#[derive(Debug, Clone, Copy, Default)]
pub struct TacticalOpponent;

impl<const R: usize, const C: usize> OpponentPolicy<R, C> for TacticalOpponent {
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
            if let Ok(row) = clone.drop_piece(col) {
                if clone.check_win_at(row, col, me) {
                    return col;
                }
            }
        }

        // 2. Check for immediate blocking move
        for &col in &legal {
            let mut clone = s.clone();
            clone.current_player = opp;
            if let Ok(row) = clone.drop_piece(col) {
                if clone.check_win_at(row, col, opp) {
                    return col;
                }
            }
        }

        // 3. Prefer center column or closest to center
        let center = C / 2;
        *legal
            .iter()
            .min_by_key(|&&col| (col as isize - center as isize).abs())
            .unwrap()
    }
}

/// Macro-action planning dynamics executing complete game rounds.
///
/// Each transition step executes:
/// 1. The primary agent's action.
/// 2. If non-terminal, the opponent's reaction chosen by `opponent_policy`.
///
/// Decision nodes in MCTS using this dynamics are **strictly agent-centric** (always primary agent to move).
#[derive(Debug, Clone)]
pub struct MacroConnect4Dynamics<P, const R: usize = 6, const C: usize = 7> {
    /// Canonical referee managing the game state.
    pub world: Connect4World<R, C>,
    /// Policy governing opponent reactions.
    pub opponent_policy: P,
    /// Color of the primary planning agent.
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
    P: OpponentPolicy<R, C>,
{
    type State = Connect4State<R, C>;
    type Action = usize;
    type Reward = [f32; 2];

    #[inline]
    fn initial(&self) -> Self::State {
        self.world.initial()
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        self.world.legal_actions(s, out);
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        // 1. Apply primary agent's move
        let primary_outcome = self.world.step_action(s, *action);
        if primary_outcome.terminated {
            return primary_outcome;
        }

        // 2. Opponent replies via pluggable policy
        let opp_action = self.opponent_policy.select_action(s);
        self.world.step_action(s, opp_action)
    }

    #[inline]
    fn current_agent(&self, _s: &Self::State) -> AgentId {
        AgentId(0) // In macro dynamics, it is always the primary agent's turn to decide
    }
}

impl<P, const R: usize, const C: usize> BatchedAgentDynamics for MacroConnect4Dynamics<P, R, C>
where
    P: OpponentPolicy<R, C>,
{
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_outcomes);
    }
}
