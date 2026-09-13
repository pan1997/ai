//! Planning dynamics adapters for Hex search agents.

use crate::world::HexWorld;
use mcts_traits::TurnBasedDynamics;

/// Single-agent hypothetical transition dynamics for Hex on an $N \times N$ board.
///
/// Wraps [`HexWorld<N>`] in [`TurnBasedDynamics`] to provide [`mcts_traits::AgentDynamics`]
/// and [`mcts_traits::BatchedAgentDynamics`] with zero allocation on hot paths.
pub type HexDynamics<const N: usize = 11> = TurnBasedDynamics<HexWorld<N>>;

/// Constructs standard Hex planning dynamics for an $N \times N$ board.
#[inline]
pub fn new_dynamics<const N: usize>() -> HexDynamics<N> {
    TurnBasedDynamics::new(HexWorld::<N>::new())
}

/// Constructs Hex planning dynamics for an $N \times N$ board with optional Pie rule.
#[inline]
pub fn new_dynamics_with_pie_rule<const N: usize>(pie_rule: bool) -> HexDynamics<N> {
    TurnBasedDynamics::new(HexWorld::<N>::with_pie_rule(pie_rule))
}
