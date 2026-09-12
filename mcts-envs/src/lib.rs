//! # MCTS Envs
//!
//! Reference games, benchmark environments, and baseline evaluators for Monte Carlo Tree Search.
//!
//! ## Reference Environments
//!
//! - [`hex`]: Hex board game with parametric dimensions $N \times N$, disjoint-set union-find (DSU)
//!   connectivity win detection, implementing [`World`](mcts_traits::World) and [`TurnBasedWorld`](mcts_traits::TurnBasedWorld).
//! - [`tzf8`]: 2048 single-player stochastic puzzle with tile merges and xorshift random tile spawning.
//! - [`kuhn_poker`]: Kuhn Poker 3-card imperfect-information game demonstrating private card deals, public pot history,
//!   and the separation between [`World`](mcts_traits::World) and [`AgentDynamics`](mcts_traits::AgentDynamics).
//!
//! Note: Connect 4 is now located in its own dedicated workspace crate: `connect4`.
//!
//! ## Baseline Evaluators
//!
//! - [`evaluators::RolloutEvaluator`]: Classical Monte Carlo uniform random rollouts to terminal states or depth limit.
//! - [`evaluators::UniformRandomModel`]: Fast baseline providing uniform priors and zero values for testing search flow.

pub mod evaluators;
pub mod hex;
pub mod kuhn_poker;
pub mod tzf8;

#[cfg(test)]
mod tests;

pub use evaluators::{RolloutEvaluator, UniformRandomModel};
pub use hex::{HexPlayer, HexState, HexWorld};
pub use kuhn_poker::{KuhnAction, KuhnAgentDynamics, KuhnObservation, KuhnWorld, KuhnWorldState};
pub use tzf8::{Direction, Tzf8Dynamics, Tzf8State};
