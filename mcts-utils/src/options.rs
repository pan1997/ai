//! Visualization styling and traversal options for MCTS tree rendering.

use mcts_engine::tree_store::{NodeId as MctsNodeId, NodeStatus};
use mcts_traits::AgentId;
use std::sync::Arc;

/// Graph layout direction for Graphviz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RankDir {
    /// Top-to-bottom layout (standard vertical hierarchy).
    #[default]
    TopToBottom,
    /// Left-to-right layout (horizontal sequence).
    LeftToRight,
}

impl RankDir {
    /// Returns the Graphviz DOT attribute value (e.g. `"TB"` or `"LR"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopToBottom => "TB",
            Self::LeftToRight => "LR",
        }
    }
}

/// Predicate function closure determining whether a node should be included given `(node_id, visits)`.
pub type NodePredicate = Arc<dyn Fn(MctsNodeId, u32) -> bool + Send + Sync>;

/// Formatter function closure mapping an action reference to a display string.
pub type ActionFormatter<Action> = Arc<dyn Fn(&Action) -> String + Send + Sync>;

/// Formatter function closure mapping a step delta reference to a display string.
pub type DeltaFormatter<StepDelta> = Arc<dyn Fn(&StepDelta) -> String + Send + Sync>;

/// Formatter function closure producing an optional custom node label.
pub type CustomNodeLabelFn =
    Arc<dyn Fn(MctsNodeId, NodeStatus, AgentId) -> Option<String> + Send + Sync>;

/// Node filtering criteria for focusing visualization on relevant search paths.
#[derive(Clone, Default)]
pub enum NodeFilter {
    /// Render all reachable nodes in the tree (default).
    #[default]
    All,
    /// Render only nodes that have at least `min_visits` incoming visits (root is always included).
    MinVisits(u32),
    /// Render only the top `N` nodes ordered by visit count (root is always included).
    TopNodes(usize),
    /// Custom predicate closure determining whether a node should be included given `(node_id, visits)`.
    Custom(NodePredicate),
}

/// Traversal, formatting, and aesthetic options for MCTS tree rendering.
#[derive(Clone)]
pub struct TreeVisualizerOptions<Action, StepDelta = ()> {
    /// Graph identifier name in the DOT output.
    pub graph_id: String,
    /// Orientation of tree layout (Top-to-bottom or Left-to-right).
    pub rankdir: RankDir,
    /// Maximum search depth to include in the visual tree (None = full depth).
    pub max_depth: Option<usize>,
    /// Minimum edge visits required to include an edge in the visualization.
    pub min_visits: u32,
    /// Node filtering strategy (e.g. all nodes, top N nodes, or min node visits).
    pub node_filter: NodeFilter,
    /// Whether to highlight the Principal Variation (PV / highest-visit path).
    pub highlight_pv: bool,
    /// Font family for graph nodes and edges.
    pub font_name: String,
    /// Whether to display policy prior values $P(s, a)$ on edges.
    pub show_priors: bool,
    /// Whether to display Q-value estimates on edges.
    pub show_q_values: bool,
    /// Whether to display availability counts $N_{\text{avail}}$ on edges (for ISMCTS).
    pub show_avail_visits: bool,
    /// Whether to display immediate transition rewards on edges.
    pub show_rewards: bool,
    /// Whether to render intermediate virtual chance nodes for stochastic action transitions.
    pub render_chance_nodes: bool,
    /// Optional closure for custom action label formatting.
    pub action_formatter: Option<ActionFormatter<Action>>,
    /// Optional closure for custom transition delta formatting.
    pub delta_formatter: Option<DeltaFormatter<StepDelta>>,
    /// Optional closure for custom node label formatting.
    pub custom_node_label: Option<CustomNodeLabelFn>,
}

impl<Action, StepDelta> Default for TreeVisualizerOptions<Action, StepDelta> {
    fn default() -> Self {
        Self {
            graph_id: "mcts_tree".to_string(),
            rankdir: RankDir::TopToBottom,
            max_depth: None,
            min_visits: 0,
            node_filter: NodeFilter::All,
            highlight_pv: true,
            font_name: "Helvetica,Arial,sans-serif".to_string(),
            show_priors: true,
            show_q_values: true,
            show_avail_visits: true,
            show_rewards: true,
            render_chance_nodes: false,
            action_formatter: None,
            delta_formatter: None,
            custom_node_label: None,
        }
    }
}

impl<Action, StepDelta> TreeVisualizerOptions<Action, StepDelta> {
    /// Creates a new `TreeVisualizerOptions` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures whether to render intermediate virtual chance nodes for stochastic action transitions.
    #[must_use]
    pub fn with_render_chance_nodes(mut self, enabled: bool) -> Self {
        self.render_chance_nodes = enabled;
        self
    }

    /// Sets the Graphviz graph identifier.
    #[must_use]
    pub fn with_graph_id(mut self, graph_id: impl Into<String>) -> Self {
        self.graph_id = graph_id.into();
        self
    }

    /// Sets the layout orientation to left-to-right or top-to-bottom.
    #[must_use]
    pub fn with_rankdir(mut self, rankdir: RankDir) -> Self {
        self.rankdir = rankdir;
        self
    }

    /// Sets a maximum depth cutoff for rendering large search trees.
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets a minimum edge visit threshold to prune unvisited or low-visit branches.
    #[must_use]
    pub fn with_min_visits(mut self, min_visits: u32) -> Self {
        self.min_visits = min_visits;
        self
    }

    /// Sets a node filtering strategy (e.g. [`NodeFilter::TopNodes`] or [`NodeFilter::MinVisits`]).
    #[must_use]
    pub fn with_node_filter(mut self, filter: NodeFilter) -> Self {
        self.node_filter = filter;
        self
    }

    /// Convenience builder to retain only the top `n` nodes ranked by visit count (root always included).
    #[must_use]
    pub fn with_top_nodes(mut self, n: usize) -> Self {
        self.node_filter = NodeFilter::TopNodes(n);
        self
    }

    /// Convenience builder to retain only nodes with at least `min_visits` (root always included).
    #[must_use]
    pub fn with_min_node_visits(mut self, min_visits: u32) -> Self {
        self.node_filter = NodeFilter::MinVisits(min_visits);
        self
    }

    /// Configures whether to visually highlight the Principal Variation (PV).
    #[must_use]
    pub fn with_highlight_pv(mut self, highlight: bool) -> Self {
        self.highlight_pv = highlight;
        self
    }

    /// Configures a custom action formatter closure.
    #[must_use]
    pub fn with_action_formatter<F>(mut self, formatter: F) -> Self
    where
        F: Fn(&Action) -> String + Send + Sync + 'static,
    {
        self.action_formatter = Some(Arc::new(formatter));
        self
    }

    /// Configures a custom transition delta formatter closure.
    #[must_use]
    pub fn with_delta_formatter<F>(mut self, formatter: F) -> Self
    where
        F: Fn(&StepDelta) -> String + Send + Sync + 'static,
    {
        self.delta_formatter = Some(Arc::new(formatter));
        self
    }

    /// Configures a custom node label formatter closure.
    #[must_use]
    pub fn with_custom_node_label<F>(mut self, formatter: F) -> Self
    where
        F: Fn(MctsNodeId, NodeStatus, AgentId) -> Option<String> + Send + Sync + 'static,
    {
        self.custom_node_label = Some(Arc::new(formatter));
        self
    }
}
