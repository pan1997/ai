use super::{BackupPolicy, MultiAgentReward, PathElement};
use crate::selection::MultiAgentPuctStats;
use crate::tree_store::{EdgeId, NodeId, NodeStatus, PriorStore, TreeStore};
use mcts_traits::{HasPolicy, HasValue};

/// Single-agent discounted backup strategy ($N = 1$).
///
/// Accumulates scalar discounted returns backwards along the path:
///
/// $$G_t = r(s_t, a_t) + \gamma G_{t+1}$$
///
/// And updates running edge means incrementally:
///
/// $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{G_t - Q(s_t, a_t)}{N(s_t, a_t)}$$
///
/// Ideal for single-player environments (e.g. 2048), MuZero latent dynamics, and belief-state determinizations.
pub struct SingleAgentBackup {
    /// Discount factor $\gamma \in [0, 1]$ applied to future values.
    pub gamma: f32,
}

impl Default for SingleAgentBackup {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

impl SingleAgentBackup {
    /// Creates a new `SingleAgentBackup` with the specified discount factor $\gamma$.
    pub fn new(gamma: f32) -> Self {
        Self { gamma }
    }
}

impl<A, R, Eval, StepDelta> BackupPolicy<A, R, MultiAgentPuctStats<1>, Eval, StepDelta>
    for SingleAgentBackup
where
    A: Clone,
    R: Copy + MultiAgentReward<1>,
    Eval: HasValue + HasPolicy,
{
    fn init_root(
        &self,
        store: &mut TreeStore<A, R, MultiAgentPuctStats<1>, StepDelta>,
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
        store: &mut TreeStore<A, R, MultiAgentPuctStats<1>, StepDelta>,
        path: &[PathElement],
        evaluation: Option<&Eval>,
    ) {
        if path.is_empty() {
            return;
        }

        let last_element = path.last().unwrap();
        let leaf_node = last_element.next_node;

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
