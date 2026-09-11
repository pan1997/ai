use super::{MultiAgentPuctStats, SelectionPolicy};
use crate::tree_store::{EdgeId, NodeId, TreeStore};

/// Gumbel AlphaZero selection.
///
/// In Gumbel MCTS (Danihelka et al., 2022):
/// - Gumbel noise is applied to the policy logits at the ROOT node to guarantee policy improvement.
/// - Child / interior nodes use deterministic PUCT selection to avoid variance accumulation.
pub struct GumbelPuctSelection<const N: usize> {
    pub c_puct: f32,
}

impl<const N: usize> Default for GumbelPuctSelection<N> {
    fn default() -> Self {
        Self { c_puct: 1.0 }
    }
}

impl<Action, Reward, const N: usize> SelectionPolicy<Action, Reward, MultiAgentPuctStats<N>>
    for GumbelPuctSelection<N>
{
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, MultiAgentPuctStats<N>>,
        node_id: NodeId,
    ) -> Option<EdgeId> {
        let active_agent = store.node_agent(node_id).0 as usize;
        assert!(
            active_agent < N,
            "GumbelPuctSelection: active agent ID ({active_agent}) must be in range 0..{N}"
        );

        let first = store.first_child_edge(node_id);
        let count = store.num_children(node_id);
        if count == 0 {
            return None;
        }

        let is_root = !store.parent_edge(node_id).is_valid();

        let parent_visits: u32 = if is_root {
            store
                .child_edges(node_id)
                .map(|e| store.stats.visits[e.as_usize()])
                .sum()
        } else {
            store.stats.visits[store.parent_edge(node_id).as_usize()]
        };
        let parent_visits_sqrt = (parent_visits as f32).sqrt();

        let mut best_edge = EdgeId::INVALID;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let edge_idx = edge.as_usize();

            let visits = store.stats.visits[edge_idx];
            let q = store.stats.mean_value[edge_idx][active_agent];
            let prior = store.stats.priors[edge_idx];
            let v_loss = store.stats.virtual_loss[edge_idx];

            let effective_visits = visits as f32 + v_loss;
            let effective_q = if effective_visits > 0.0 {
                (q * (visits as f32) - v_loss) / effective_visits
            } else {
                0.0
            };

            let score = if is_root {
                // At the root node: apply Gumbel perturbation to logits
                let mut hash = (node_id.as_usize() as u32)
                    .wrapping_mul(314159265)
                    .wrapping_add((edge.as_usize() as u32).wrapping_mul(271828182));
                if hash == 0 {
                    hash = 1;
                }
                hash ^= hash << 13;
                hash ^= hash >> 17;
                hash ^= hash << 5;
                let u = (hash as f64 / (u32::MAX as f64 + 1.0)) as f32;
                let u_clamped = u.clamp(1e-7, 0.999999);
                let gumbel = -(-u_clamped.ln()).ln();
                let logit = (prior + 1e-7).ln();

                effective_q
                    + self.c_puct * (parent_visits_sqrt / (1.0 + effective_visits)) * (logit + gumbel)
            } else {
                // At interior nodes: standard deterministic PUCT (no Gumbel noise)
                let u = self.c_puct * prior * parent_visits_sqrt / (1.0 + effective_visits);
                effective_q + u
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

