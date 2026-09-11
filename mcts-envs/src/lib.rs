//! # MCTS Envs
//!
//! Reference games, benchmark environments, and baseline evaluators for Monte Carlo Tree Search.
//!
//! ## Reference Environments
//!
//! - [`connect4`]: Connect 4 with parametric dimensions $R \times C$ (default: $6 \times 7$), implementing
//!   [`AgentDynamics`](mcts_traits::AgentDynamics), [`BatchedAgentDynamics`](mcts_traits::BatchedAgentDynamics),
//!   and [`World`](mcts_traits::World).
//! - [`hex`]: Hex board game with parametric dimensions $N \times N$, graph connectivity win detection via BFS,
//!   implementing [`AgentDynamics`](mcts_traits::AgentDynamics) and [`World`](mcts_traits::World).
//! - [`tzf8`]: 2048 single-player stochastic puzzle with tile merges and xorshift random tile spawning.
//! - [`kuhn_poker`]: Kuhn Poker 3-card imperfect-information game demonstrating private card deals, public pot history,
//!   and the separation between [`World`](mcts_traits::World) and [`AgentDynamics`](mcts_traits::AgentDynamics).
//!
//! ## Baseline Evaluators
//!
//! - [`evaluators::RolloutEvaluator`]: Classical Monte Carlo uniform random rollouts to terminal states or depth limit.
//! - [`evaluators::UniformRandomModel`]: Fast baseline providing uniform priors and zero values for testing search flow.

pub mod connect4;
pub mod hex;
pub mod tzf8;
pub mod kuhn_poker;
pub mod evaluators;

#[cfg(test)]
mod tests;

pub use connect4::{Connect4Dynamics, Connect4State, Player};
pub use hex::{HexDynamics, HexState, HexPlayer};
pub use tzf8::{Tzf8Dynamics, Tzf8State, Direction};
pub use kuhn_poker::{KuhnAction, KuhnAgentDynamics, KuhnObservation, KuhnWorld, KuhnWorldState};
pub use evaluators::{RolloutEvaluator, UniformRandomModel};
