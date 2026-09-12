//! Min-Max normalized selection policies for scale-invariant MCTS on arbitrary reward domains.
//!
//! Classical UCT and AlphaZero PUCT assume returns $Q \in [0, 1]$ or $Q \in [-1, 1]$.
//! When applied to environments with large, unbounded, or non-zero-sum returns
//! (such as 2048 where scores reach tens of thousands), raw $Q$ values can dwarf
//! the exploration bonus, causing search to lock onto the first evaluated branch.
//!
//! The policies in this module implement dynamic Min-Max sibling normalization (originating
//! from Kocsis & Szepesvári and popularized by DeepMind's MuZero), mapping running
//! values $Q(s, a)$ dynamically to $[0, 1]$:
//!
//! $$Q_{\text{norm}}(s, a) = \frac{Q_i(s, a) - Q_{\min}}{Q_{\max} - Q_{\min}}$$
//!
//! This preserves the theoretical exploration/exploitation balance regardless of whether
//! game returns are measured in fractions, thousands, or millions.

use super::{MultiAgentPuctStats, SelectionPolicy};
use crate::tree_store::{EdgeId, NodeId, TreeStore};

/// Min-Max Normalized Upper Confidence Bounds for Trees (UCT) selection policy.
///
/// Normalizes running mean values $Q(s, a)$ among visited sibling edges to $[0, 1]$:
///
/// $$Q_{\text{norm}}(s, a) = \frac{Q_i(s, a) - Q_{\min}}{Q_{\max} - Q_{\min}}$$
///
/// Edge scores are computed as:
///
/// $$\text{Score}(s, a) = \begin{cases} +\infty & \text{if } N(s, a) = 0 \\ Q_{\text{norm}}(s, a) + c_{\text{uct}} \sqrt{\frac{\ln(N(s) + 1)}{N(s, a)}} & \text{if } N(s, a) > 0 \end{cases}$$
pub struct NormalizedUctSelection<const N: usize> {
    /// Exploration constant scaling the confidence interval (defaults to $\sqrt{2} \approx 1.4142$).
    pub c_uct: f32,
}

impl<const N: usize> Default for NormalizedUctSelection<N> {
    fn default() -> Self {
        Self {
            c_uct: std::f32::consts::SQRT_2,
        }
    }
}

impl<const N: usize> NormalizedUctSelection<N> {
    /// Creates a new `NormalizedUctSelection` with custom exploration constant $c_{\text{uct}}$.
    pub fn new(c_uct: f32) -> Self {
        Self { c_uct }
    }
}

impl<Action, Reward, StepDelta, const N: usize>
    SelectionPolicy<Action, Reward, MultiAgentPuctStats<N>, StepDelta>
    for NormalizedUctSelection<N>
{
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, MultiAgentPuctStats<N>, StepDelta>,
        node_id: NodeId,
    ) -> Option<EdgeId> {
        let active_agent = store.node_agent(node_id).0 as usize;
        assert!(
            active_agent < N,
            "NormalizedUctSelection: active agent ID ({active_agent}) must be in range 0..{N}"
        );

        let first = store.first_child_edge(node_id);
        let count = store.num_children(node_id);
        if count == 0 {
            return None;
        }

        let mut parent_visits: u32 = 0;
        let mut q_min = f32::INFINITY;
        let mut q_max = f32::NEG_INFINITY;
        let mut unvisited_edge = EdgeId::INVALID;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let edge_idx = edge.as_usize();
            let visits = store.stats.visits[edge_idx];
            parent_visits += visits;

            if visits == 0 {
                if !unvisited_edge.is_valid() {
                    unvisited_edge = edge;
                }
            } else {
                let q = store.stats.mean_value[edge_idx][active_agent];
                if q < q_min {
                    q_min = q;
                }
                if q > q_max {
                    q_max = q;
                }
            }
        }

        // 1. Unvisited children get +infinity score, visited first
        if unvisited_edge.is_valid() {
            return Some(unvisited_edge);
        }

        let ln_parent = ((parent_visits + 1) as f32).ln();
        let q_range = q_max - q_min;

        let mut best_edge = EdgeId::INVALID;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let edge_idx = edge.as_usize();

            let visits = store.stats.visits[edge_idx];
            let q = store.stats.mean_value[edge_idx][active_agent];

            let q_norm = if q_range > 1e-6 {
                (q - q_min) / q_range
            } else {
                0.5
            };

            let u = self.c_uct * (ln_parent / (visits as f32)).sqrt();
            let score = q_norm + u;

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

/// Min-Max Normalized Predictor Upper Confidence Bounds for Trees (PUCT) selection policy.
///
/// Implements MuZero-style dynamic min-max normalization across visited sibling edges:
///
/// $$Q_{\text{norm}}(s, a) = \begin{cases} \frac{Q_i(s, a) - Q_{\min}}{Q_{\max} - Q_{\min}} & \text{if } N(s, a) > 0 \text{ and } Q_{\max} > Q_{\min} \\ 0.0 & \text{if } N(s, a) = 0 \\ 0.5 & \text{otherwise} \end{cases}$$
///
/// And scores candidate edges with virtual loss adjustment:
///
/// $$\text{Score}(s, a) = Q_{\text{eff\_norm}}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N_{\text{eff}}(s, a)}$$
///
/// Unvisited edges use First-Play Urgency (FPU) $Q_{\text{norm}} = 0.0$, allowing their policy
/// prior probability $P(s, a)$ to dictate the initial exploration order on equal footing
/// with normalized visited returns in $[0, 1]$.
pub struct NormalizedPuctSelection<const N: usize> {
    /// Exploration constant $c_{\text{puct}}$ scaling the influence of the policy prior.
    pub c_puct: f32,
}

impl<const N: usize> Default for NormalizedPuctSelection<N> {
    fn default() -> Self {
        Self {
            c_puct: std::f32::consts::SQRT_2,
        }
    }
}

impl<const N: usize> NormalizedPuctSelection<N> {
    /// Creates a new `NormalizedPuctSelection` with custom exploration constant $c_{\text{puct}}$.
    pub fn new(c_puct: f32) -> Self {
        Self { c_puct }
    }
}

impl<Action, Reward, StepDelta, const N: usize>
    SelectionPolicy<Action, Reward, MultiAgentPuctStats<N>, StepDelta>
    for NormalizedPuctSelection<N>
{
    fn select_child(
        &self,
        store: &TreeStore<Action, Reward, MultiAgentPuctStats<N>, StepDelta>,
        node_id: NodeId,
    ) -> Option<EdgeId> {
        let active_agent = store.node_agent(node_id).0 as usize;
        assert!(
            active_agent < N,
            "NormalizedPuctSelection: active agent ID ({active_agent}) must be in range 0..{N}"
        );

        let first = store.first_child_edge(node_id);
        let count = store.num_children(node_id);
        if count == 0 {
            return None;
        }

        let mut parent_visits: u32 = 0;
        let mut visited_count: u32 = 0;
        let mut q_min = f32::INFINITY;
        let mut q_max = f32::NEG_INFINITY;
        let mut best_prior_edge = EdgeId::INVALID;
        let mut max_prior = f32::NEG_INFINITY;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let edge_idx = edge.as_usize();
            let visits = store.stats.visits[edge_idx];
            parent_visits += visits;

            let prior = store.stats.priors[edge_idx];
            if prior > max_prior {
                max_prior = prior;
                best_prior_edge = edge;
            }

            if visits > 0 {
                visited_count += 1;
                let q = store.stats.mean_value[edge_idx][active_agent];
                if q < q_min {
                    q_min = q;
                }
                if q > q_max {
                    q_max = q;
                }
            }
        }

        // When no children have been visited yet, choose the child with the highest prior
        if visited_count == 0 {
            return if best_prior_edge.is_valid() {
                Some(best_prior_edge)
            } else {
                Some(first)
            };
        }

        let parent_visits_sqrt = (parent_visits as f32).sqrt();
        let q_range = q_max - q_min;

        let mut best_edge = EdgeId::INVALID;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..count {
            let edge = EdgeId(first.0 + i);
            let edge_idx = edge.as_usize();

            let visits = store.stats.visits[edge_idx];
            let prior = store.stats.priors[edge_idx];
            let v_loss = store.stats.virtual_loss[edge_idx];

            let effective_visits = visits as f32 + v_loss;

            let effective_q = if visits > 0 {
                let q = store.stats.mean_value[edge_idx][active_agent];
                let q_norm = if q_range > 1e-6 {
                    (q - q_min) / q_range
                } else {
                    0.5
                };
                if effective_visits > 0.0 {
                    (q_norm * (visits as f32) - v_loss) / effective_visits
                } else {
                    0.0
                }
            } else {
                // FPU: unvisited edges default to 0.0 normalized value
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
