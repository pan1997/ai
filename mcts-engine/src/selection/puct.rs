use super::SelectionPolicy;
use crate::tree_store::{EdgeId, EdgeStatsStore, NodeId, PriorStore, TreeStore, VirtualLossStore};

/// Statistics storage for multi-agent AlphaZero-style PUCT selection.
///
/// Tracks visit counts, policy priors, running mean return vectors for $N$ agents,
/// and virtual loss weights for batched search.
#[derive(Debug, Clone)]
pub struct MultiAgentPuctStats<const N: usize> {
    /// Number of times each edge has been traversed during search.
    pub visits: Vec<u32>,
    /// Policy prior probability $P(s, a)$ assigned to each edge by the evaluation model.
    pub priors: Vec<f32>,
    /// Running mean value estimates $\mathbf{Q}(s, a) \in \mathbb{R}^N$ for all agents.
    pub mean_value: Vec<[f32; N]>,
    /// Temporary virtual loss weight accumulated during batched parallel traversals.
    pub virtual_loss: Vec<f32>,
}

impl<const N: usize> MultiAgentPuctStats<N> {
    /// Creates a new, empty `MultiAgentPuctStats` container.
    pub fn new() -> Self {
        Self {
            visits: Vec::new(),
            priors: Vec::new(),
            mean_value: Vec::new(),
            virtual_loss: Vec::new(),
        }
    }
}

impl<const N: usize> Default for MultiAgentPuctStats<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> EdgeStatsStore for MultiAgentPuctStats<N> {
    fn resize(&mut self, new_len: usize) {
        self.visits.resize(new_len, 0);
        self.priors.resize(new_len, 0.0);
        self.mean_value.resize(new_len, [0.0; N]);
        self.virtual_loss.resize(new_len, 0.0);
    }

    fn clear(&mut self) {
        self.visits.clear();
        self.priors.clear();
        self.mean_value.clear();
        self.virtual_loss.clear();
    }

    fn retain_edges(&mut self, kept_indices: &[usize]) {
        self.visits = kept_indices.iter().map(|&i| self.visits[i]).collect();
        self.priors = kept_indices.iter().map(|&i| self.priors[i]).collect();
        self.mean_value = kept_indices.iter().map(|&i| self.mean_value[i]).collect();
        self.virtual_loss = kept_indices.iter().map(|&i| self.virtual_loss[i]).collect();
    }
}

impl<const N: usize> PriorStore for MultiAgentPuctStats<N> {
    fn prior(&self, edge: EdgeId) -> f32 {
        self.priors[edge.as_usize()]
    }

    fn set_prior(&mut self, edge: EdgeId, prior: f32) {
        self.priors[edge.as_usize()] = prior;
    }
}

impl<const N: usize> VirtualLossStore for MultiAgentPuctStats<N> {
    fn add_virtual_loss(&mut self, edge: EdgeId, weight: f32) {
        self.virtual_loss[edge.as_usize()] += weight;
    }

    fn remove_virtual_loss(&mut self, edge: EdgeId, weight: f32) {
        self.virtual_loss[edge.as_usize()] -= weight;
    }
}

/// Predictor Upper Confidence Bounds for Trees (PUCT) selection policy with virtual loss.
///
/// Computes edge scores using the AlphaZero formulation adjusted for active agent perspective and virtual loss:
///
/// $$\text{Score}(s, a) = Q_{\text{eff}}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N_{\text{eff}}(s, a)}$$
///
/// Where:
/// - $N_{\text{eff}}(s, a) = N(s, a) + v_{\text{loss}}(s, a)$
/// - $Q_{\text{eff}}(s, a) = \frac{Q_i(s, a) \cdot N(s, a) - v_{\text{loss}}(s, a)}{N_{\text{eff}}(s, a)}$ (if $N_{\text{eff}} > 0$)
/// - $i$ is the active agent at node $s$.
pub struct MultiAgentPuctSelection<const N: usize> {
    /// Exploration constant $c_{\text{puct}}$ scaling the influence of the prior policy distribution.
    pub c_puct: f32,
}

impl<Action, Reward, StepDelta, const N: usize>
    SelectionPolicy<Action, Reward, MultiAgentPuctStats<N>, StepDelta>
    for MultiAgentPuctSelection<N>
{
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, MultiAgentPuctStats<N>, StepDelta>,
        node_id: NodeId,
    ) -> Option<EdgeId> {
        let active_agent = store.node_agent(node_id).0 as usize;
        assert!(
            active_agent < N,
            "MultiAgentPuctSelection: active agent ID ({active_agent}) must be in range 0..{N}"
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

            let u = self.c_puct * prior * parent_visits_sqrt / (1.0 + effective_visits);
            let score = effective_q + u;

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
