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

pub mod arena;
pub mod backup;
pub mod dirichlet;
pub mod opponent;
pub mod scheduler;
pub mod search;
pub mod selection;
pub mod tree_store;

#[cfg(test)]
mod tests;

pub use arena::{
    GameOutcome, H2HMatrix, MultiPlayerTournamentStats, TwoPlayerTournamentStats,
    disambiguate_names,
};
pub use backup::{BackupPolicy, MultiAgentReward, PathElement, SingleAgentBackup, VectorBackup};
pub use dirichlet::{add_dirichlet_noise, add_root_dirichlet_noise};
pub use opponent::{AdversarialOpponent, HeuristicOpponent, RandomOpponent, TreeOpponentPolicy};
pub use scheduler::{BatchedScheduler, MultiGameScheduler, SequentialScheduler};
pub use search::{TrajectoryOutcome, descend_trajectory};
pub use selection::{
    GumbelPuctSelection, MultiAgentPuctSelection, MultiAgentPuctStats, NormalizedPuctSelection,
    NormalizedUctSelection, SelectionPolicy, UctSelection,
};
pub use tree_store::{
    EdgeId, EdgeStatsStore, NodeId, NodeStatus, PriorStore, TreeStore, VirtualLossStore,
};
