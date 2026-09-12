//! Value backpropagation and return accumulation strategies.
//!
//! After a leaf state is evaluated or identified as terminal, a backup policy propagates returns
//! backward along the traversed root-to-leaf path, updating visit counts and running value estimates.

pub mod single;
pub mod vector;

pub use single::SingleAgentBackup;
pub use vector::VectorBackup;

use crate::tree_store::{EdgeId, EdgeStatsStore, NodeId, TreeStore};

/// Represents an element in the search path traversed during selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathElement {
    /// The parent node traversed.
    pub node: NodeId,
    /// The child edge chosen from `node`.
    pub edge: EdgeId,
    /// The target child node reached by traversing `edge` and its step delta.
    pub next_node: NodeId,
}

/// Interface for extracting per-agent rewards for $N$ players.
pub trait MultiAgentReward<const N: usize> {
    /// Returns the reward vector for all $N$ agents.
    fn agent_rewards(&self) -> [f32; N];
}

impl<const N: usize> MultiAgentReward<N> for [f32; N] {
    #[inline]
    fn agent_rewards(&self) -> [f32; N] {
        *self
    }
}

/// Policy interface for backpropagating evaluations along traversed trajectories.
pub trait BackupPolicy<Action, Reward, Stats: EdgeStatsStore, Evaluation, StepDelta = ()> {
    /// Initializes statistics for the root node during search setup (e.g. writing priors).
    fn init_root(
        &self,
        store: &mut TreeStore<Action, Reward, Stats, StepDelta>,
        root: NodeId,
        evaluation: &Evaluation,
    );

    /// Backpropagates the leaf evaluation result along the search path.
    ///
    /// If `evaluation` is `None`, the leaf is treated as terminal with zero subsequent value.
    fn backup(
        &self,
        store: &mut TreeStore<Action, Reward, Stats, StepDelta>,
        path: &[PathElement],
        evaluation: Option<&Evaluation>,
    );
}
