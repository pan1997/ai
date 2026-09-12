use super::{BackupPolicy, MultiAgentReward, PathElement};
use crate::selection::MultiAgentPuctStats;
use crate::tree_store::{EdgeId, NodeId, NodeStatus, PriorStore, TreeStore};
use mcts_traits::{HasPolicy, HasValue};

/// Vector backup strategy across $N$ agents.
///
/// Accumulates discounted returns element-wise for all agents:
///
/// $$\mathbf{G}_t = \mathbf{r}(s_t, a_t) + \gamma \mathbf{G}_{t+1}$$
///
/// And incrementally updates running mean vector estimates $\mathbf{Q}(s_t, a_t) \in \mathbb{R}^N$:
///
/// $$\mathbf{Q}(s_t, a_t) \leftarrow \mathbf{Q}(s_t, a_t) + \frac{\mathbf{G}_t - \mathbf{Q}(s_t, a_t)}{N(s_t, a_t)}$$
///
/// Completely avoids the negation convention bug class ($Q \leftarrow -Q$), naturally handles
/// non-zero-sum games, and preserves game-theoretic invariants across any player count $N$.
pub struct VectorBackup<const N: usize> {
    /// Discount factor $\gamma \in [0, 1]$ applied element-wise to future value vectors.
    pub gamma: f32,
}

impl<const N: usize> Default for VectorBackup<N> {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

impl<const N: usize> VectorBackup<N> {
    /// Creates a new `VectorBackup` policy with the specified discount factor $\gamma$.
    pub fn new(gamma: f32) -> Self {
        Self { gamma }
    }

    /// Writes normalized priors to outgoing child edges of a node.
    fn write_priors<A, R, Eval, StepDelta>(
        store: &mut TreeStore<A, R, MultiAgentPuctStats<N>, StepDelta>,
        node: NodeId,
        evaluation: &Eval,
    ) where
        Eval: HasPolicy,
    {
        let policy_priors = evaluation.priors();
        let num_children = store.num_children(node);
        let first_edge = store.first_child_edge(node);
        if num_children == 0 {
            return;
        }

        assert_eq!(
            policy_priors.len(),
            num_children as usize,
            "VectorBackup: prior probabilities length ({}) must match legal children count ({num_children})",
            policy_priors.len()
        );

        let sum: f32 = policy_priors.iter().sum();
        if sum > 0.0 {
            for i in 0..num_children {
                let edge = EdgeId(first_edge.0 + i);
                let normalized = policy_priors[i as usize] / sum;
                store.stats.set_prior(edge, normalized);
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

impl<A, R, Eval, StepDelta, const N: usize>
    BackupPolicy<A, R, MultiAgentPuctStats<N>, Eval, StepDelta> for VectorBackup<N>
where
    A: Clone,
    R: Copy + MultiAgentReward<N>,
    Eval: HasValue + HasPolicy,
{
    fn init_root(
        &self,
        store: &mut TreeStore<A, R, MultiAgentPuctStats<N>, StepDelta>,
        root: NodeId,
        evaluation: &Eval,
    ) {
        let root_agent = store.node_agent(root).0 as usize;
        assert!(
            root_agent < N,
            "VectorBackup::init_root: root agent ID ({root_agent}) must be in range 0..{N}"
        );
        if store.node_status(root) == NodeStatus::Expanded {
            Self::write_priors(store, root, evaluation);
        }
    }

    fn backup(
        &self,
        store: &mut TreeStore<A, R, MultiAgentPuctStats<N>, StepDelta>,
        path: &[PathElement],
        evaluation: Option<&Eval>,
    ) {
        if path.is_empty() {
            return;
        }

        let last_element = path.last().unwrap();
        let leaf_node = last_element.next_node;

        let leaf_agent = store.node_agent(leaf_node).0 as usize;
        assert!(
            leaf_agent < N,
            "VectorBackup::backup: leaf agent ID ({leaf_agent}) must be in range 0..{N}"
        );

        // Initialize priors on leaf outgoing edges if expanded and non-terminal
        if let (NodeStatus::Expanded, Some(eval)) = (store.node_status(leaf_node), evaluation) {
            Self::write_priors(store, leaf_node, eval);
        }

        // Initialize general return vector with leaf values (0.0 for all agents if terminal)
        let mut g = [0.0; N];
        if let Some(eval) = evaluation {
            let values = eval.values();
            assert_eq!(
                values.len(),
                N,
                "VectorBackup: evaluation values length ({}) must match N ({N})",
                values.len()
            );
            g.copy_from_slice(values);
        }

        // Accumulate returns backwards along path
        for i in (0..path.len()).rev() {
            let element = &path[i];
            let edge_reward = store
                .edge_reward(element.edge)
                .expect("VectorBackup: transition reward must be set for traversed path edges")
                .agent_rewards();

            for (g_val, &reward_val) in g.iter_mut().zip(edge_reward.iter()) {
                *g_val = reward_val + self.gamma * (*g_val);
            }

            let edge_idx = element.edge.as_usize();
            let old_visits = store.stats.visits[edge_idx];
            let new_visits = old_visits + 1;
            store.stats.visits[edge_idx] = new_visits;

            for (q_val, &g_val) in store.stats.mean_value[edge_idx].iter_mut().zip(g.iter()) {
                let q_old = *q_val;
                *q_val = q_old + (g_val - q_old) / (new_visits as f32);
            }
        }
    }
}
