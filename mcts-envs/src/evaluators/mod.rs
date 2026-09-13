//! Baseline heuristic and rollout evaluation models.
//!
//! Provides reference implementations of [`Model`](mcts_traits::Model) for benchmarking
//! environments without requiring deep neural networks:
//! - [`RolloutEvaluator`]: Monte Carlo random playouts to terminal states or max depth.
//! - [`UniformRandomModel`]: Unbiased uniform prior distribution with zero value estimates.

/// Baseline uniform random evaluation model.
pub mod random;
/// Monte Carlo rollout simulation evaluation model.
pub mod rollout;

pub use random::UniformRandomModel;
pub use rollout::RolloutEvaluator;
