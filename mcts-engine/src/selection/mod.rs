//! Tree traversal and child selection strategies.
//!
//! Selection policies define how the tree search descends from the root to an unexpanded or
//! leaf node by picking the most promising child edge according to exploration/exploitation trade-offs.

pub mod puct;
pub mod uct;
pub mod gumbel;

pub use puct::{MultiAgentPuctSelection, MultiAgentPuctStats};
pub use uct::UctSelection;
pub use gumbel::GumbelPuctSelection;

use crate::tree_store::{TreeStore, NodeId, EdgeId, EdgeStatsStore};

/// Selection policy interface for choosing child edges during search descent.
pub trait SelectionPolicy<Action, Reward, Stats: EdgeStatsStore> {
    /// Selects the best child edge originating from `node_id` in `store`.
    ///
    /// Returns `Some(EdgeId)` if the node has legal child edges, or `None` if it has no children.
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, Stats>,
        node_id: NodeId,
    ) -> Option<EdgeId>;
}

