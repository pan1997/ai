//! # MCTS Engine
//!
//! High-performance, zero-allocation Structure-of-Arrays (SoA) Monte Carlo Tree Search engine.
//!
//! ## Overview
//!
//! `mcts-engine` implements cache-conscious MCTS using flat array allocations and typed 32-bit
//! index handles ([`NodeId`], [`EdgeId`]). It supports modern deep reinforcement learning and
//! game-theoretic planning algorithms, including:
//! - **Selection**: Classic UCT, AlphaZero-style PUCT with virtual loss, and Gumbel AlphaZero.
//! - **Backup**: Discounted single-agent returns ($Q = r + \gamma V$) and $N$-player vector returns ($Q \in \mathbb{R}^N$).
//! - **Schedulers**: Single-threaded sequential sweeps, virtual-loss batched GPU simulation, and vectorized multi-game self-play.
//!
//! ## Core Submodules
//!
//! - [`tree_store`]: The contiguous Structure-of-Arrays [`TreeStore`], node/edge arrays, and statistics traits.
//! - [`selection`]: Policy interfaces and algorithms ([`UctSelection`], [`MultiAgentPuctSelection`], [`GumbelPuctSelection`]).
//! - [`backup`]: Value backpropagation policies ([`SingleAgentBackup`], [`VectorBackup`]).
//! - [`scheduler`]: Execution engines ([`SequentialScheduler`], [`BatchedScheduler`], [`MultiGameScheduler`]).

pub mod tree_store;
pub mod selection;
pub mod backup;
pub mod scheduler;
pub mod dirichlet;

#[cfg(test)]
mod tests;

pub use tree_store::{
    EdgeId, EdgeStatsStore, NodeId, NodeStatus, PriorStore, TreeStore, VirtualLossStore,
};
pub use selection::{
    GumbelPuctSelection, MultiAgentPuctSelection, MultiAgentPuctStats, SelectionPolicy,
    UctSelection,
};
pub use backup::{
    BackupPolicy, MultiAgentReward, PathElement, SingleAgentBackup, VectorBackup,
};
pub use scheduler::{BatchedScheduler, MultiGameScheduler, SequentialScheduler};
pub use dirichlet::{add_dirichlet_noise, add_root_dirichlet_noise};
