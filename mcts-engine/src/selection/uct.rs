use super::{MultiAgentPuctStats, SelectionPolicy};
use crate::tree_store::{EdgeId, NodeId, TreeStore};

/// Classic Upper Confidence Bounds for Trees (UCT) selection policy.
///
/// Balances exploitation of high-value actions with exploration of rarely visited actions:
///
/// $$\text{Score}(s, a) = \begin{cases} +\infty & \text{if } N(s, a) = 0 \\ Q_i(s, a) + c_{\text{uct}} \sqrt{\frac{\ln(N(s) + 1)}{N(s, a)}} & \text{if } N(s, a) > 0 \end{cases}$$
///
/// Where:
/// - $N(s, a)$ is the visit count of child edge $a$.
/// - $N(s)$ is the total visit count of parent node $s$.
/// - $Q_i(s, a)$ is the running mean value for active agent $i$.
/// - $c_{\text{uct}}$ is the exploration constant (defaults to $\sqrt{2} \approx 1.4142$).
pub struct UctSelection<const N: usize> {
    /// Exploration constant scaling the confidence interval.
    pub c_uct: f32,
}

impl<const N: usize> Default for UctSelection<N> {
    fn default() -> Self {
        Self {
            c_uct: std::f32::consts::SQRT_2,
        }
    }
}

impl<Action, Reward, StepDelta, const N: usize>
    SelectionPolicy<Action, Reward, MultiAgentPuctStats<N>, StepDelta> for UctSelection<N>
{
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, MultiAgentPuctStats<N>, StepDelta>,
        node_id: NodeId,
    ) -> Option<EdgeId> {
        let active_agent = store.node_agent(node_id).0 as usize;
        assert!(
            active_agent < N,
            "UctSelection: active agent ID ({active_agent}) must be in range 0..{N}"
        );

        let first = store.first_child_edge(node_id);
        let count = store.num_children(node_id);
        if count == 0 {
            return None;
        }

        let parent_visits: u32 = if store.parent_edge(node_id).is_valid() {
            store.stats.visits[store.parent_edge(node_id).as_usize()]
        } else {
            store
                .child_edges(node_id)
                .map(|e| store.stats.visits[e.as_usize()])
                .sum()
        };

        let ln_parent = ((parent_visits + 1) as f32).ln();

        let mut best_edge = EdgeId::INVALID;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let edge_idx = edge.as_usize();

            let visits = store.stats.visits[edge_idx];
            let q = store.stats.mean_value[edge_idx][active_agent];

            let score = if visits == 0 {
                f32::INFINITY
            } else {
                q + self.c_uct * (ln_parent / visits as f32).sqrt()
            };

            if score > best_score {
                best_score = score;
                best_edge = edge;
            }
        }

        if best_edge.is_valid() {
            Some(best_edge)
        } else {
            None
        }
    }
}
