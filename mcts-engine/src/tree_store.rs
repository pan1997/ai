use mcts_traits::AgentId;

/// Strongly typed 32-bit handle representing a node in the [`TreeStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Sentinel value indicating an invalid or uninitialized node.
    pub const INVALID: Self = Self(u32::MAX);

    /// Returns `true` if this handle refers to a valid node index (i.e. not [`Self::INVALID`]).
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    /// Converts the node handle to a `usize` for array indexing.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Strongly typed 32-bit handle representing an outgoing edge/action in the [`TreeStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

impl EdgeId {
    /// Sentinel value indicating an invalid or uninitialized edge.
    pub const INVALID: Self = Self(u32::MAX);

    /// Returns `true` if this handle refers to a valid edge index (i.e. not [`Self::INVALID`]).
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    /// Converts the edge handle to a `usize` for array indexing.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Expansion lifecycle status of a tree search node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeStatus {
    /// Node is newly created; legal action edges have not yet been generated.
    #[default]
    Unexpanded,
    /// Node has been expanded; its child edges are populated in the tree store.
    Expanded,
    /// Node is a terminal game state with no further legal actions.
    Terminal,
}

/// Storage interface for edge-associated statistics (visits, values, etc.).
pub trait EdgeStatsStore {
    /// Resizes internal statistics arrays to accommodate `new_len` total edges.
    fn resize(&mut self, new_len: usize);

    /// Clears all statistics, resetting them to an empty state.
    fn clear(&mut self);
}

/// Interface for statistics stores that record policy prior probabilities on edges.
pub trait PriorStore {
    /// Sets the policy prior probability for `edge`.
    fn set_prior(&mut self, edge: EdgeId, prior: f32);
}

/// Interface for statistics stores that support virtual loss tracking for parallel or batched search.
pub trait VirtualLossStore {
    /// Temporarily increments the virtual loss on `edge` by `weight`.
    fn add_virtual_loss(&mut self, edge: EdgeId, weight: f32);

    /// Reverts previously added virtual loss from `edge` by `weight`.
    fn remove_virtual_loss(&mut self, edge: EdgeId, weight: f32);
}

/// Structure-of-Arrays (SoA) memory layout for high cache locality and zero-allocation search traversals.
///
/// Stores tree nodes and outgoing edges in flat, contiguous vectors, indexed by 32-bit [`NodeId`]
/// and [`EdgeId`]. This design guarantees sequential memory access patterns when iterating over
/// sibling child edges, minimizing CPU cache misses and eliminating heap allocation overhead
/// along the critical search path.
pub struct TreeStore<Action, Reward, Stats: EdgeStatsStore> {
    nodes: NodeArrays,
    edges: EdgeArrays<Action, Reward>,
    /// Attached statistics storage (e.g. visit counts, running value means, priors, virtual loss).
    pub stats: Stats,
}

struct NodeArrays {
    parent_edge: Vec<EdgeId>,
    first_child_edge: Vec<EdgeId>,
    num_children: Vec<u32>,
    agent: Vec<AgentId>,
    status: Vec<NodeStatus>,
}

struct EdgeArrays<Action, Reward> {
    action: Vec<Action>,
    child_node: Vec<NodeId>,
    reward: Vec<Option<Reward>>,
}

impl<Action, Reward, Stats: EdgeStatsStore> TreeStore<Action, Reward, Stats> {
    /// Returns the total number of nodes currently allocated in the tree.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.parent_edge.len()
    }

    /// Returns the total number of edges currently allocated in the tree.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.action.len()
    }

    /// Returns the lifecycle expansion status of `node`.
    #[inline]
    pub fn node_status(&self, node: NodeId) -> NodeStatus {
        self.nodes.status[node.as_usize()]
    }

    /// Returns `true` if `node` has already been expanded with legal actions.
    #[inline]
    pub fn is_expanded(&self, node: NodeId) -> bool {
        self.nodes.status[node.as_usize()] == NodeStatus::Expanded
    }

    /// Returns the handle of the first child edge of `node`.
    #[inline]
    pub fn first_child_edge(&self, node: NodeId) -> EdgeId {
        self.nodes.first_child_edge[node.as_usize()]
    }

    /// Returns the number of legal action child edges emanating from `node`.
    #[inline]
    pub fn num_children(&self, node: NodeId) -> u32 {
        self.nodes.num_children[node.as_usize()]
    }

    /// Returns the edge handle that leads into `node` from its parent (or [`EdgeId::INVALID`] if root).
    #[inline]
    pub fn parent_edge(&self, node: NodeId) -> EdgeId {
        self.nodes.parent_edge[node.as_usize()]
    }

    /// Returns the identifier of the agent whose turn it is to act at `node`.
    #[inline]
    pub fn node_agent(&self, node: NodeId) -> AgentId {
        self.nodes.agent[node.as_usize()]
    }

    /// Returns the child node handle pointed to by `edge` (or [`NodeId::INVALID`] if unexpanded).
    #[inline]
    pub fn edge_child(&self, edge: EdgeId) -> NodeId {
        self.edges.child_node[edge.as_usize()]
    }

    /// Returns a reference to the action associated with `edge`.
    #[inline]
    pub fn edge_action(&self, edge: EdgeId) -> &Action {
        &self.edges.action[edge.as_usize()]
    }

    /// Returns an optional reference to the immediate transition reward associated with `edge`.
    #[inline]
    pub fn edge_reward(&self, edge: EdgeId) -> Option<&Reward> {
        self.edges.reward[edge.as_usize()].as_ref()
    }

    /// Returns an iterator yielding all child edge handles for `node` in contiguous order.
    #[inline]
    pub fn child_edges(&self, node: NodeId) -> impl Iterator<Item = EdgeId> {
        let first = self.first_child_edge(node);
        let count = self.num_children(node);
        (0..count).map(move |i| EdgeId(first.0 + i))
    }
}

impl<Action: Clone, Reward: Clone, Stats: EdgeStatsStore> TreeStore<Action, Reward, Stats> {
    /// Pre-allocates a `TreeStore` with specified capacities for nodes and edges.
    ///
    /// Sizing capacities appropriately prevents dynamic vector reallocations during tree expansion.
    pub fn with_capacity(nodes_cap: usize, edges_cap: usize, stats: Stats) -> Self {
        Self {
            nodes: NodeArrays {
                parent_edge: Vec::with_capacity(nodes_cap),
                first_child_edge: Vec::with_capacity(nodes_cap),
                num_children: Vec::with_capacity(nodes_cap),
                agent: Vec::with_capacity(nodes_cap),
                status: Vec::with_capacity(nodes_cap),
            },
            edges: EdgeArrays {
                action: Vec::with_capacity(edges_cap),
                child_node: Vec::with_capacity(edges_cap),
                reward: Vec::with_capacity(edges_cap),
            },
            stats,
        }
    }

    /// Resets the tree store, clearing all nodes, edges, and statistics.
    pub fn clear(&mut self) {
        self.nodes.parent_edge.clear();
        self.nodes.first_child_edge.clear();
        self.nodes.num_children.clear();
        self.nodes.agent.clear();
        self.nodes.status.clear();

        self.edges.action.clear();
        self.edges.child_node.clear();
        self.edges.reward.clear();

        self.stats.clear();
    }

    /// Inserts the initial root node for a new search tree with active player `agent`.
    ///
    /// Returns the allocated [`NodeId`] (typically `NodeId(0)`).
    pub fn insert_root(&mut self, agent: AgentId) -> NodeId {
        let node_id = NodeId(self.nodes.parent_edge.len() as u32);
        self.nodes.parent_edge.push(EdgeId::INVALID);
        self.nodes.first_child_edge.push(EdgeId::INVALID);
        self.nodes.num_children.push(0);
        self.nodes.agent.push(agent);
        self.nodes.status.push(NodeStatus::Unexpanded);
        node_id
    }

    /// Allocates a new unexpanded child node attached to `parent_edge`.
    ///
    /// # Panics
    ///
    /// Panics if `parent_edge` is out of bounds or already has a linked child node.
    pub fn insert_node(&mut self, parent_edge: EdgeId, agent: AgentId) -> NodeId {
        assert!(
            parent_edge.as_usize() < self.edges.action.len(),
            "insert_node: parent_edge out of bounds"
        );
        assert_eq!(
            self.edges.child_node[parent_edge.as_usize()],
            NodeId::INVALID,
            "insert_node: parent_edge already has a linked child node"
        );

        let node_id = NodeId(self.nodes.parent_edge.len() as u32);
        self.nodes.parent_edge.push(parent_edge);
        self.nodes.first_child_edge.push(EdgeId::INVALID);
        self.nodes.num_children.push(0);
        self.nodes.agent.push(agent);
        self.nodes.status.push(NodeStatus::Unexpanded);

        self.edges.child_node[parent_edge.as_usize()] = node_id;
        node_id
    }

    /// Expands `node` by allocating contiguous edges for all provided `actions`.
    ///
    /// Transitions the node status to [`NodeStatus::Expanded`] and resizes edge statistics storage.
    ///
    /// # Panics
    ///
    /// Panics if `node` is not in [`NodeStatus::Unexpanded`] state.
    pub fn expand_node(&mut self, node: NodeId, actions: &[Action]) -> EdgeId {
        assert_eq!(
            self.nodes.status[node.as_usize()],
            NodeStatus::Unexpanded,
            "expand_node: node must be Unexpanded"
        );

        let first_edge_idx = self.edges.action.len() as u32;
        let num_actions = actions.len();

        self.edges.action.extend_from_slice(actions);
        self.edges
            .child_node
            .resize(self.edges.child_node.len() + num_actions, NodeId::INVALID);
        self.edges
            .reward
            .resize(self.edges.reward.len() + num_actions, None);

        self.stats.resize(self.edges.action.len());

        let edge_id = EdgeId(first_edge_idx);
        self.nodes.first_child_edge[node.as_usize()] = edge_id;
        self.nodes.num_children[node.as_usize()] = num_actions as u32;
        self.nodes.status[node.as_usize()] = NodeStatus::Expanded;

        edge_id
    }

    /// Marks `node` as terminal, preventing further expansion attempts.
    ///
    /// # Panics
    ///
    /// Panics if the node was already expanded.
    pub fn mark_terminal(&mut self, node: NodeId) {
        let status = self.nodes.status[node.as_usize()];
        assert!(
            status == NodeStatus::Unexpanded || status == NodeStatus::Terminal,
            "mark_terminal: node must be Unexpanded or Terminal"
        );
        self.nodes.status[node.as_usize()] = NodeStatus::Terminal;
    }

    /// Stores the transition reward on `edge`.
    ///
    /// # Panics
    ///
    /// Panics if `edge` is out of bounds or a reward has already been assigned to this edge.
    pub fn set_edge_reward(&mut self, edge: EdgeId, reward: Reward) {
        assert!(
            edge.as_usize() < self.edges.action.len(),
            "set_edge_reward: edge ID out of bounds"
        );
        assert!(
            self.edges.reward[edge.as_usize()].is_none(),
            "set_edge_reward: reward has already been set for this edge"
        );
        self.edges.reward[edge.as_usize()] = Some(reward);
    }
}

