pub mod puct;
pub mod uct;
pub mod gumbel;

pub use puct::{MultiAgentPuctSelection, MultiAgentPuctStats};
pub use uct::UctSelection;
pub use gumbel::GumbelPuctSelection;

use crate::tree_store::{TreeStore, NodeId, EdgeId, EdgeStatsStore};

pub trait SelectionPolicy<Action, Reward, Stats: EdgeStatsStore> {
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, Stats>,
        node_id: NodeId,
    ) -> Option<EdgeId>;
}

