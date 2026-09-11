//! # MCTS Traits
//!
//! Universal, zero-dependency abstractions for Monte Carlo Tree Search (MCTS), AlphaZero,
//! MuZero, and Game AI in Rust.
//!
//! ## Overview
//!
//! `mcts-traits` defines minimal, unopinionated trait interfaces governing state transitions,
//! evaluation models, multi-agent refereeing, and identity representations. It avoids imposing
//! blanket bounds (such as `Clone`, `Send`, or `Sync`) on associated types, enabling search
//! algorithms to dictate their own requirements without constraining underlying domain models.
//!
//! ## Core Abstractions
//!
//! - [`AgentDynamics`]: Single-agent transition model used during hypothetical tree search planning.
//! - [`BatchedAgentDynamics`]: High-throughput vectorized dynamics for GPU/SIMD-accelerated environments.
//! - [`Model`]: Prior and value evaluation interface (e.g. neural networks or heuristic rollouts).
//! - [`BatchedModel`]: Vectorized evaluation for amortizing deep neural network tensor inference.
//! - [`World`]: External environment referee managing impartial ground-truth state, multi-player observation filtering, and simultaneous moves.
//! - [`AgentId`]: Strongly typed 32-bit player/agent identifier.
//! - [`GraphEnv`]: Deterministic state machine graph for exact mathematical testing of search mechanics.
//!
//! ## Module Organization
//!
//! - [`agent`]: Agent identity representation ([`AgentId`]).
//! - [`dynamics`]: State transitions ([`Transition`], [`AgentDynamics`], [`BatchedAgentDynamics`]).
//! - [`model`]: Policy priors and value estimators ([`Model`], [`BatchedModel`], [`Evaluation`]).
//! - [`world`]: Ground-truth referee arbitration ([`World`]).
//! - [`graph`]: Unit testing environment harness ([`GraphEnv`]).

pub mod agent;
pub mod dynamics;
pub mod model;
pub mod world;
pub mod graph;

#[cfg(test)]
mod tests;

pub use agent::AgentId;
pub use dynamics::{
    default_step_batch, AgentDynamics, BatchedAgentDynamics, StepOutcome, Transition,
};
pub use model::{BatchedModel, Evaluation, HasPolicy, HasValue, Model};
pub use world::World;
pub use graph::GraphEnv;
