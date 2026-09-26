//! MCTS search schedulers and execution orchestrators.
//!
//! Schedulers govern how simulation passes are scheduled, evaluated, and backpropagated:
//! - [`SequentialScheduler`]: Standard 1-by-1 root-to-leaf sweeps.
//! - [`BatchedScheduler`]: Parallel simulation within a single tree using virtual loss and leaf deduplication.
//! - [`MultiGameScheduler`]: Vectorized search across multiple disjoint game trees for batched self-play.

/// Virtual-loss-guided batched leaf evaluation scheduler.
pub mod batched;
/// Single-Tree Information Set MCTS (Single-Tree ISMCTS) scheduler.
pub mod ismcts;
/// Vectorized search across multiple disjoint game trees.
pub mod multi_game;
/// Leaf deduplication and expansion scheduling utilities for batched searches.
pub mod dedup;
/// Single-threaded depth-first sequential scheduler.
pub mod sequential;

pub use batched::BatchedScheduler;
pub use dedup::{ExpansionRequest, LeafDeduplicator};
pub use ismcts::IsmctsScheduler;
pub use multi_game::MultiGameScheduler;
pub use sequential::SequentialScheduler;
