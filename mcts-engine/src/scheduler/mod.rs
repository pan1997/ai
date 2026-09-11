//! MCTS search schedulers and execution orchestrators.
//!
//! Schedulers govern how simulation passes are scheduled, evaluated, and backpropagated:
//! - [`SequentialScheduler`]: Standard 1-by-1 root-to-leaf sweeps.
//! - [`BatchedScheduler`]: Parallel simulation within a single tree using virtual loss and leaf deduplication.
//! - [`MultiGameScheduler`]: Vectorized search across multiple disjoint game trees for batched self-play.

pub mod sequential;
pub mod batched;
pub mod multi_game;

pub use sequential::SequentialScheduler;
pub use batched::BatchedScheduler;
pub use multi_game::MultiGameScheduler;

