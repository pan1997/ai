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
//! Note: Production-grade games with dedicated CLI players and arenas are located in their own workspace crates:
//! - `connect4`: Dedicated Connect 4 engine, MCTS agents, and tournament CLI.
//! - `blokus`: Dedicated Blokus (Classic & Duo) engine, polyomino registry, and tournament CLI.
//! - `tzf8`: Dedicated 2048 bitboard engine, Expectimax agents, ANSI renderer, and tournament arena.
//! - `hex`: Dedicated Hex engine, DSU connectivity tracking, shortest-path heuristic, and tournament arena.
//!
//! (The [`tzf8`] and [`hex`] modules in this crate are lightweight reference re-exports for test suites).
//!
//! ## Baseline Evaluators
//!
//! - [`evaluators::RolloutEvaluator`]: Classical Monte Carlo uniform random rollouts to terminal states or depth limit.
//! - [`evaluators::UniformRandomModel`]: Fast baseline providing uniform priors and zero values for testing search flow.

pub mod evaluators;
pub mod hex;
/// Kuhn Poker imperfect-information game theory benchmark.
pub mod kuhn_poker;
pub mod tzf8;

#[cfg(test)]
mod tests;

pub use evaluators::{RolloutEvaluator, UniformRandomModel};
pub use hex::{HexPlayer, HexState, HexWorld};
pub use kuhn_poker::{KuhnAction, KuhnAgentDynamics, KuhnObservation, KuhnWorld, KuhnWorldState};
pub use tzf8::{Direction, Tzf8Dynamics, Tzf8State};
