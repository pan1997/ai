//! Tree traversal and child selection strategies.
//!
//! Selection policies define how the tree search descends from the root to an unexpanded or
//! leaf node by picking the most promising child edge according to exploration/exploitation trade-offs.
//!
//! Supported policies include:
//! - [`UctSelection`]: Classic Upper Confidence Bounds for Trees (Kocsis & Szepesvári, 2006).
//! - [`NormalizedUctSelection`]: Dynamic sibling Min-Max normalized UCT for arbitrary reward/score scales.
//! - [`MultiAgentPuctSelection`]: AlphaZero-style Predictor UCT with virtual loss support.
//! - [`NormalizedPuctSelection`]: MuZero-style dynamic Min-Max normalized PUCT with First Play Urgency (FPU).
//! - [`GumbelPuctSelection`]: Danihelka et al. (2022) policy improvement via Gumbel noise at the root.

/// Danihelka et al. (2022) Gumbel AlphaZero policy improvement selection.
pub mod gumbel;
/// Dynamic Min-Max sibling score normalization selection policies (MuZero-style).
pub mod normalized;
/// AlphaZero-style Predictor Upper Confidence Bounds for Trees (PUCT) selection.
pub mod puct;
/// Classic Upper Confidence Bounds for Trees (UCT) selection.
pub mod uct;

pub use gumbel::GumbelPuctSelection;
pub use normalized::{NormalizedPuctSelection, NormalizedUctSelection};
pub use puct::{MultiAgentPuctSelection, MultiAgentPuctStats};
pub use uct::UctSelection;

use crate::tree_store::{EdgeId, EdgeStatsStore, NodeId, TreeStore};

/// Selection policy interface for choosing child edges during search descent.
pub trait SelectionPolicy<Action, Reward, Stats: EdgeStatsStore, StepDelta = ()> {
    /// Selects the best child edge originating from `node_id` in `store`.
    ///
    /// Returns `Some(EdgeId)` if the node has legal child edges, or `None` if it has no children.
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, Stats, StepDelta>,
        node_id: NodeId,
    ) -> Option<EdgeId>;
}
