//! Edge statistics extraction and formatting for MCTS tree visualization.

use mcts_engine::selection::MultiAgentPuctStats;
use mcts_engine::selection::ismcts::IsmctsStats;
use mcts_engine::tree_store::EdgeId;

/// Interface for inspecting edge statistics during tree graph rendering.
pub trait EdgeStatsView {
    /// Number of times `edge` was visited/traversed.
    fn visits(&self, edge: EdgeId) -> u32;

    /// Number of times `edge` was available when its parent was reached (for ISMCTS).
    fn avail_visits(&self, _edge: EdgeId) -> Option<u32> {
        None
    }

    /// Policy prior probability $P(s, a)$ on `edge`, if recorded.
    fn prior(&self, _edge: EdgeId) -> Option<f32> {
        None
    }

    /// Scalar or primary agent Q-value on `edge`.
    fn primary_q(&self, _edge: EdgeId) -> Option<f32> {
        None
    }

    /// Formatted statistics string for display on graph edges.
    fn format_stats(&self, edge: EdgeId) -> String;
}

impl<const N: usize> EdgeStatsView for MultiAgentPuctStats<N> {
    #[inline]
    fn visits(&self, edge: EdgeId) -> u32 {
        self.visits.get(edge.as_usize()).copied().unwrap_or(0)
    }

    #[inline]
    fn prior(&self, edge: EdgeId) -> Option<f32> {
        self.priors.get(edge.as_usize()).copied()
    }

    #[inline]
    fn primary_q(&self, edge: EdgeId) -> Option<f32> {
        self.mean_value
            .get(edge.as_usize())
            .map(|arr| arr.first().copied().unwrap_or(0.0))
    }

    fn format_stats(&self, edge: EdgeId) -> String {
        let idx = edge.as_usize();
        let visits = self.visits.get(idx).copied().unwrap_or(0);
        let prior = self.priors.get(idx).copied().unwrap_or(0.0);
        let q = self.mean_value.get(idx).copied().unwrap_or([0.0; N]);

        if N == 1 {
            format!("N: {}\nQ: {:.3}\nP: {:.3}", visits, q[0], prior)
        } else {
            let q_str = q
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>()
                .join(", ");
            format!("N: {}\nQ: [{}]\nP: {:.3}", visits, q_str, prior)
        }
    }
}

impl<const N: usize> EdgeStatsView for IsmctsStats<N> {
    #[inline]
    fn visits(&self, edge: EdgeId) -> u32 {
        self.visits.get(edge.as_usize()).copied().unwrap_or(0)
    }

    #[inline]
    fn avail_visits(&self, edge: EdgeId) -> Option<u32> {
        self.avail_visits.get(edge.as_usize()).copied()
    }

    #[inline]
    fn prior(&self, edge: EdgeId) -> Option<f32> {
        self.priors.get(edge.as_usize()).copied()
    }

    #[inline]
    fn primary_q(&self, edge: EdgeId) -> Option<f32> {
        self.mean_value
            .get(edge.as_usize())
            .map(|arr| arr.first().copied().unwrap_or(0.0))
    }

    fn format_stats(&self, edge: EdgeId) -> String {
        let idx = edge.as_usize();
        let visits = self.visits.get(idx).copied().unwrap_or(0);
        let avail = self.avail_visits.get(idx).copied().unwrap_or(0);
        let prior = self.priors.get(idx).copied().unwrap_or(0.0);
        let q = self.mean_value.get(idx).copied().unwrap_or([0.0; N]);

        if N == 1 {
            format!(
                "N: {} (avail: {})\nQ: {:.3}\nP: {:.3}",
                visits, avail, q[0], prior
            )
        } else {
            let q_str = q
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "N: {} (avail: {})\nQ: [{}]\nP: {:.3}",
                visits, avail, q_str, prior
            )
        }
    }
}

impl EdgeStatsView for () {
    #[inline]
    fn visits(&self, _edge: EdgeId) -> u32 {
        0
    }

    #[inline]
    fn format_stats(&self, _edge: EdgeId) -> String {
        String::new()
    }
}
