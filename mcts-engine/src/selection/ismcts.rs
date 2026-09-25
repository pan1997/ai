//! Information Set Monte Carlo Tree Search (Single-Tree ISMCTS) selection and statistics storage.

use super::SelectionPolicy;
use crate::tree_store::{EdgeId, EdgeStatsStore, NodeId, PriorStore, TreeStore, VirtualLossStore};

/// Structure-of-Arrays (SoA) statistics storage specifically designed for Single-Tree ISMCTS.
///
/// In addition to standard traversed visit counts $N(s, a)$, policy priors $P(s, a)$, and running mean
/// vectors $\mathbf{Q}(s, a) \in \mathbb{R}^N$, this store tracks availability visit counts $N_{\text{avail}}(s, a)$.
///
/// An action $a$ has its availability count incremented whenever node $s$ is reached in a simulation
/// trajectory and action $a$ is legally compatible with that trajectory's sampled state determinization.
#[derive(Debug, Clone)]
pub struct IsmctsStats<const N: usize> {
    /// Number of times each edge has been selected and traversed along a simulation trajectory.
    pub visits: Vec<u32>,
    /// Number of times each edge was legally available when its parent node was visited.
    pub avail_visits: Vec<u32>,
    /// Policy prior probability $P(s, a)$ assigned to each edge by the evaluation model.
    pub priors: Vec<f32>,
    /// Running mean value return vectors $\mathbf{Q}(s, a) \in \mathbb{R}^N$ for all agents.
    pub mean_value: Vec<[f32; N]>,
    /// Temporary virtual loss weight accumulated during batched parallel traversals.
    pub virtual_loss: Vec<f32>,
}

impl<const N: usize> IsmctsStats<N> {
    /// Creates a new, empty `IsmctsStats` container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            visits: Vec::new(),
            avail_visits: Vec::new(),
            priors: Vec::new(),
            mean_value: Vec::new(),
            virtual_loss: Vec::new(),
        }
    }

    /// Returns the availability visit count $N_{\text{avail}}(s, a)$ for `edge`.
    #[inline]
    #[must_use]
    pub fn avail_visits(&self, edge: EdgeId) -> u32 {
        self.avail_visits[edge.as_usize()]
    }

    /// Increments the availability visit count for `edge` by 1.
    #[inline]
    pub fn inc_avail_visits(&mut self, edge: EdgeId) {
        self.avail_visits[edge.as_usize()] += 1;
    }

    /// Sets the availability visit count for `edge`.
    #[inline]
    pub fn set_avail_visits(&mut self, edge: EdgeId, val: u32) {
        self.avail_visits[edge.as_usize()] = val;
    }
}

impl<const N: usize> Default for IsmctsStats<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> EdgeStatsStore for IsmctsStats<N> {
    fn resize(&mut self, new_len: usize) {
        self.visits.resize(new_len, 0);
        self.avail_visits.resize(new_len, 0);
        self.priors.resize(new_len, 0.0);
        self.mean_value.resize(new_len, [0.0; N]);
        self.virtual_loss.resize(new_len, 0.0);
    }

    fn clear(&mut self) {
        self.visits.clear();
        self.avail_visits.clear();
        self.priors.clear();
        self.mean_value.clear();
        self.virtual_loss.clear();
    }

    fn retain_edges(&mut self, kept_indices: &[usize]) {
        self.visits = kept_indices.iter().map(|&i| self.visits[i]).collect();
        self.avail_visits = kept_indices.iter().map(|&i| self.avail_visits[i]).collect();
        self.priors = kept_indices.iter().map(|&i| self.priors[i]).collect();
        self.mean_value = kept_indices.iter().map(|&i| self.mean_value[i]).collect();
        self.virtual_loss = kept_indices.iter().map(|&i| self.virtual_loss[i]).collect();
    }
}

impl<const N: usize> VirtualLossStore for IsmctsStats<N> {
    #[inline]
    fn add_virtual_loss(&mut self, edge: EdgeId, weight: f32) {
        self.virtual_loss[edge.as_usize()] += weight;
    }

    #[inline]
    fn remove_virtual_loss(&mut self, edge: EdgeId, weight: f32) {
        self.virtual_loss[edge.as_usize()] -= weight;
    }
}

impl<const N: usize> PriorStore for IsmctsStats<N> {
    #[inline]
    fn prior(&self, edge: EdgeId) -> f32 {
        self.priors[edge.as_usize()]
    }

    #[inline]
    fn set_prior(&mut self, edge: EdgeId, prior: f32) {
        self.priors[edge.as_usize()] = prior;
    }
}

/// Availability-weighted Predictor Upper Confidence Bounds for Trees (ISMCTS PUCT) selection.
///
/// Implements the availability-weighted selection formula for Information Set MCTS (Cowling et al., 2012):
///
/// $$\text{Score}(s, a) = Q_{\text{eff}}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N_{\text{avail}}(s)}}{1 + N_{\text{eff}}(s, a)}$$
///
/// Where:
/// - $N_{\text{eff}}(s, a) = N(s, a) + v_{\text{loss}}(s, a)$.
/// - $Q_{\text{eff}}(s, a) = \frac{Q_i(s, a) \cdot N(s, a) - v_{\text{loss}}(s, a)}{N_{\text{eff}}(s, a)}$ (if $N_{\text{eff}} > 0$), else $0.0$.
/// - $N_{\text{avail}}(s) = \sum_{a' \in \mathcal{A}_{\text{avail}}(s)} N_{\text{avail}}(s, a')$ is the total availability mass across legal candidate edges.
/// - $i$ is the active agent at node $s$.
#[derive(Debug, Clone, Copy)]
pub struct IsmctsSelection<const N: usize> {
    /// Exploration constant scaling the influence of availability-weighted policy priors.
    pub c_puct: f32,
}

impl<const N: usize> Default for IsmctsSelection<N> {
    fn default() -> Self {
        Self { c_puct: 1.414 }
    }
}

impl<const N: usize> IsmctsSelection<N> {
    /// Constructs a new `IsmctsSelection` with the given exploration constant.
    #[must_use]
    pub const fn new(c_puct: f32) -> Self {
        Self { c_puct }
    }

    /// Selects the best child edge originating from `node_id` among those that satisfy `is_compatible`.
    ///
    /// Computes the availability score across all candidate child edges for which `is_compatible(edge, action)` is true.
    /// Returns `None` if the node has no children or if no child edges are compatible with the current state.
    pub fn select_compatible_child<Action, Reward, StepDelta, F>(
        &self,
        store: &TreeStore<Action, Reward, IsmctsStats<N>, StepDelta>,
        node_id: NodeId,
        is_compatible: F,
    ) -> Option<EdgeId>
    where
        F: Fn(EdgeId, &Action) -> bool,
    {
        let active_agent = store.node_agent(node_id).0 as usize;
        assert!(
            active_agent < N,
            "IsmctsSelection: active agent ID ({active_agent}) must be in range 0..{N}"
        );

        let first = store.first_child_edge(node_id);
        let count = store.num_children(node_id);
        if count == 0 {
            return None;
        }

        // 1. Calculate total availability mass across compatible edges
        let mut total_avail = 0u32;
        let mut any_compatible = false;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let action = store.edge_action(edge);
            if is_compatible(edge, action) {
                any_compatible = true;
                total_avail += store.stats.avail_visits[edge.as_usize()];
            }
        }

        if !any_compatible {
            return None;
        }

        let avail_sqrt = (total_avail.max(1) as f32).sqrt();

        // 2. Find compatible edge maximizing availability-weighted PUCT score
        let mut best_edge = EdgeId::INVALID;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let action = store.edge_action(edge);
            if !is_compatible(edge, action) {
                continue;
            }

            let edge_idx = edge.as_usize();
            let visits = store.stats.visits[edge_idx];
            let v_loss = store.stats.virtual_loss[edge_idx];
            let eff_visits = visits as f32 + v_loss;

            let q_raw = store.stats.mean_value[edge_idx][active_agent];
            let q_eff = if eff_visits > 0.0 {
                (q_raw * visits as f32 - v_loss) / eff_visits
            } else {
                0.0
            };

            let prior = store.stats.priors[edge_idx];
            let explore = self.c_puct * prior * (avail_sqrt / (1.0 + eff_visits));
            let score = q_eff + explore;

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

impl<Action, Reward, StepDelta, const N: usize>
    SelectionPolicy<Action, Reward, IsmctsStats<N>, StepDelta> for IsmctsSelection<N>
{
    #[inline]
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, IsmctsStats<N>, StepDelta>,
        node_id: NodeId,
    ) -> Option<EdgeId> {
        self.select_compatible_child(store, node_id, |_, _| true)
    }
}
