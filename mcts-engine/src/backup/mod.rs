pub mod vector;
pub mod single;

pub use vector::VectorBackup;
pub use single::SingleAgentBackup;

use crate::tree_store::{EdgeId, EdgeStatsStore, NodeId, TreeStore};

/// Represents an element in the search path traversed during selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathElement {
    pub node: NodeId,
    pub edge: EdgeId,
}

pub trait MultiAgentReward<const N: usize> {
    fn agent_rewards(&self) -> [f32; N];
}

impl<const N: usize> MultiAgentReward<N> for [f32; N] {
    #[inline]
    fn agent_rewards(&self) -> [f32; N] {
        *self
    }
}

pub trait BackupPolicy<Action, Reward, Stats: EdgeStatsStore, Evaluation> {
    /// Initializes statistics for the root node during search setup (e.g. writing priors).
    fn init_root(
        &self,
        store: &mut TreeStore<Action, Reward, Stats>,
        root: NodeId,
        evaluation: &Evaluation,
    );

    /// Backpropagates the leaf evaluation result along the search path.
    fn backup(
        &self,
        store: &mut TreeStore<Action, Reward, Stats>,
        path: &[PathElement],
        evaluation: Option<&Evaluation>,
    );
}

