use super::{BackupPolicy, MultiAgentReward, PathElement};
use crate::selection::MultiAgentPuctStats;
use crate::tree_store::{EdgeId, NodeId, NodeStatus, PriorStore, TreeStore};
use mcts_traits::{HasPolicy, HasValue};

/// Single-agent backup strategy (N=1).
///
/// Accumulates scalar discounted returns: Q = r + γ * V.
/// Ideal for single-player environments (2048), MuZero latent models, and opponent-modeled ISMCTS (Poker).
pub struct SingleAgentBackup {
    pub gamma: f32,
}

impl Default for SingleAgentBackup {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

impl SingleAgentBackup {
    pub fn new(gamma: f32) -> Self {
        Self { gamma }
    }
}

impl<A, R, Eval> BackupPolicy<A, R, MultiAgentPuctStats<1>, Eval> for SingleAgentBackup
where
    A: Clone,
    R: Copy + MultiAgentReward<1>,
    Eval: HasValue + HasPolicy,
{
    fn init_root(
        &self,
        store: &mut TreeStore<A, R, MultiAgentPuctStats<1>>,
        root: NodeId,
        evaluation: &Eval,
    ) {
        if store.node_status(root) == NodeStatus::Expanded {
            let policy_priors = evaluation.priors();
            let num_children = store.num_children(root);
            let first_edge = store.first_child_edge(root);
            if num_children == 0 {
                return;
            }

            let sum: f32 = policy_priors.iter().sum();
            if sum > 0.0 {
                for i in 0..num_children {
                    let edge = EdgeId(first_edge.0 + i);
                    store.stats.set_prior(edge, policy_priors[i as usize] / sum);
                }
            } else {
                let uniform = 1.0 / (num_children as f32);
                for i in 0..num_children {
                    let edge = EdgeId(first_edge.0 + i);
                    store.stats.set_prior(edge, uniform);
                }
            }
        }
    }

    fn backup(
        &self,
        store: &mut TreeStore<A, R, MultiAgentPuctStats<1>>,
        path: &[PathElement],
        evaluation: Option<&Eval>,
    ) {
        if path.is_empty() {
            return;
        }

        let last_element = path.last().unwrap();
        let leaf_node = store.edge_child(last_element.edge);

        if let (NodeStatus::Expanded, Some(eval)) = (store.node_status(leaf_node), evaluation) {
            let policy_priors = eval.priors();
            let num_children = store.num_children(leaf_node);
            let first_edge = store.first_child_edge(leaf_node);
            if num_children > 0 {
                let sum: f32 = policy_priors.iter().sum();
                if sum > 0.0 {
                    for i in 0..num_children {
                        let edge = EdgeId(first_edge.0 + i);
                        store.stats.set_prior(edge, policy_priors[i as usize] / sum);
                    }
                } else {
                    let uniform = 1.0 / (num_children as f32);
                    for i in 0..num_children {
                        let edge = EdgeId(first_edge.0 + i);
                        store.stats.set_prior(edge, uniform);
                    }
                }
            }
        }

        let mut g = evaluation.map(|e| e.value()).unwrap_or(0.0);

        for i in (0..path.len()).rev() {
            let element = &path[i];
            let reward = store
                .edge_reward(element.edge)
                .expect("SingleAgentBackup: transition reward must be set")
                .agent_rewards()[0];

            g = reward + self.gamma * g;

            let edge_idx = element.edge.as_usize();
            let old_visits = store.stats.visits[edge_idx];
            let new_visits = old_visits + 1;
            store.stats.visits[edge_idx] = new_visits;

            let q_old = store.stats.mean_value[edge_idx][0];
            store.stats.mean_value[edge_idx][0] = q_old + (g - q_old) / (new_visits as f32);
        }
    }
}

