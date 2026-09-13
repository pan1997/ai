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

/// Interface for statistics stores that support virtual loss tracking for parallel or batched search.
pub trait VirtualLossStore {
    /// Temporarily increments the virtual loss on `edge` by `weight`.
    ///
    /// The default implementation is a no-op for stores that do not track virtual loss.
    fn add_virtual_loss(&mut self, _edge: EdgeId, _weight: f32) {}

    /// Reverts previously added virtual loss from `edge` by `weight`.
    ///
    /// The default implementation is a no-op for stores that do not track virtual loss.
    fn remove_virtual_loss(&mut self, _edge: EdgeId, _weight: f32) {}
}

/// Storage interface for edge-associated statistics (visits, values, etc.).
pub trait EdgeStatsStore: VirtualLossStore {
    /// Resizes internal statistics arrays to accommodate `new_len` total edges.
    fn resize(&mut self, new_len: usize);

    /// Clears all statistics, resetting them to an empty state.
    fn clear(&mut self);

    /// Retains only the statistics corresponding to the edges in `kept_indices`.
    fn retain_edges(&mut self, kept_indices: &[usize]);
}

/// Interface for statistics stores that record policy prior probabilities on edges.
pub trait PriorStore {
    /// Returns the policy prior probability for `edge`.
    fn prior(&self, edge: EdgeId) -> f32;

    /// Sets the policy prior probability for `edge`.
    fn set_prior(&mut self, edge: EdgeId, prior: f32);
}

/// Structure-of-Arrays (SoA) memory layout for high cache locality and zero-allocation search traversals.
///
/// Stores tree nodes, outgoing action edges, and transition delta branches in flat, contiguous vectors,
/// indexed by 32-bit [`NodeId`] and [`EdgeId`]. This design guarantees sequential memory access patterns
/// when iterating over sibling child edges, minimizing CPU cache misses and eliminating heap allocation
/// overhead along the critical search path.
pub struct TreeStore<Action, Reward, Stats: EdgeStatsStore, StepDelta = ()> {
    nodes: NodeArrays,
    edges: EdgeArrays<Action, Reward>,
    branches: DeltaBranchArrays<StepDelta>,
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
    first_branch: Vec<u32>,
    reward: Vec<Option<Reward>>,
}

struct DeltaBranchArrays<StepDelta> {
    delta: Vec<StepDelta>,
    child_node: Vec<NodeId>,
    next_branch: Vec<u32>,
}

impl<Action, Reward, Stats: EdgeStatsStore, StepDelta> TreeStore<Action, Reward, Stats, StepDelta> {
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

    /// Returns the current capacity of allocated node storage.
    #[inline]
    pub fn node_capacity(&self) -> usize {
        self.nodes.parent_edge.capacity()
    }

    /// Returns the current capacity of allocated edge storage.
    #[inline]
    pub fn edge_capacity(&self) -> usize {
        self.edges.action.capacity()
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

    /// Returns the primary child node handle pointed to by `edge` (or [`NodeId::INVALID`] if unexpanded).
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

    /// Returns the child node handle associated with `(edge, delta)`, or `None` if not yet discovered.
    pub fn get_child(&self, edge: EdgeId, delta: &StepDelta) -> Option<NodeId>
    where
        StepDelta: PartialEq,
    {
        assert!(
            edge.as_usize() < self.edges.action.len(),
            "get_child: edge ID out of bounds"
        );
        let mut branch_idx = self.edges.first_branch[edge.as_usize()];
        while branch_idx != u32::MAX {
            let idx = branch_idx as usize;
            if &self.branches.delta[idx] == delta {
                return Some(self.branches.child_node[idx]);
            }
            branch_idx = self.branches.next_branch[idx];
        }
        None
    }

    /// Returns the child node matching `(edge, delta)`, or allocates and links a new unexpanded node.
    ///
    /// Returns `(node_id, was_newly_inserted)`.
    pub fn get_or_insert_child(
        &mut self,
        edge: EdgeId,
        delta: &StepDelta,
        agent: AgentId,
    ) -> (NodeId, bool)
    where
        StepDelta: PartialEq + Clone,
    {
        assert!(
            edge.as_usize() < self.edges.action.len(),
            "get_or_insert_child: edge ID out of bounds"
        );
        let mut branch_idx = self.edges.first_branch[edge.as_usize()];
        while branch_idx != u32::MAX {
            let idx = branch_idx as usize;
            if &self.branches.delta[idx] == delta {
                return (self.branches.child_node[idx], false);
            }
            branch_idx = self.branches.next_branch[idx];
        }

        // Allocate new unexpanded node
        let node_id = NodeId(self.nodes.parent_edge.len() as u32);
        self.nodes.parent_edge.push(edge);
        self.nodes.first_child_edge.push(EdgeId::INVALID);
        self.nodes.num_children.push(0);
        self.nodes.agent.push(agent);
        self.nodes.status.push(NodeStatus::Unexpanded);

        // Allocate new delta branch entry
        let new_branch_idx = self.branches.delta.len() as u32;
        let old_first = self.edges.first_branch[edge.as_usize()];
        self.branches.delta.push(delta.clone());
        self.branches.child_node.push(node_id);
        self.branches.next_branch.push(old_first);
        self.edges.first_branch[edge.as_usize()] = new_branch_idx;

        if self.edges.child_node[edge.as_usize()] == NodeId::INVALID {
            self.edges.child_node[edge.as_usize()] = node_id;
        }

        (node_id, true)
    }

    /// Returns an iterator yielding all `(&StepDelta, NodeId)` branches emanating from `edge`.
    pub fn delta_children(&self, edge: EdgeId) -> impl Iterator<Item = (&StepDelta, NodeId)> {
        assert!(
            edge.as_usize() < self.edges.action.len(),
            "delta_children: edge ID out of bounds"
        );
        let mut curr = self.edges.first_branch[edge.as_usize()];
        std::iter::from_fn(move || {
            if curr != u32::MAX {
                let idx = curr as usize;
                curr = self.branches.next_branch[idx];
                Some((&self.branches.delta[idx], self.branches.child_node[idx]))
            } else {
                None
            }
        })
    }
}

impl<Action: Clone, Reward: Clone, Stats: EdgeStatsStore, StepDelta: Clone>
    TreeStore<Action, Reward, Stats, StepDelta>
{
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
                first_branch: Vec::with_capacity(edges_cap),
                reward: Vec::with_capacity(edges_cap),
            },
            branches: DeltaBranchArrays {
                delta: Vec::with_capacity(edges_cap),
                child_node: Vec::with_capacity(edges_cap),
                next_branch: Vec::with_capacity(edges_cap),
            },
            stats,
        }
    }

    /// Resets the tree store, clearing all nodes, edges, delta branches, and statistics.
    pub fn clear(&mut self) {
        self.nodes.parent_edge.clear();
        self.nodes.first_child_edge.clear();
        self.nodes.num_children.clear();
        self.nodes.agent.clear();
        self.nodes.status.clear();

        self.edges.action.clear();
        self.edges.child_node.clear();
        self.edges.first_branch.clear();
        self.edges.reward.clear();

        self.branches.delta.clear();
        self.branches.child_node.clear();
        self.branches.next_branch.clear();

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

    /// Allocates a new unexpanded child node attached to `parent_edge` with default transition delta.
    ///
    /// # Panics
    ///
    /// Panics if `parent_edge` is out of bounds or already has a linked child node.
    pub fn insert_node(&mut self, parent_edge: EdgeId, agent: AgentId) -> NodeId
    where
        StepDelta: Default + PartialEq,
    {
        assert!(
            parent_edge.as_usize() < self.edges.action.len(),
            "insert_node: parent_edge out of bounds"
        );
        assert_eq!(
            self.edges.child_node[parent_edge.as_usize()],
            NodeId::INVALID,
            "insert_node: parent_edge already has a linked child node"
        );
        let (node_id, _is_new) =
            self.get_or_insert_child(parent_edge, &StepDelta::default(), agent);
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
            .first_branch
            .resize(self.edges.first_branch.len() + num_actions, u32::MAX);
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

    /// Promotes the subtree rooted at `new_root` to be the root of the search tree, discarding all unreachable nodes.
    ///
    /// Compacts reachable nodes, edges, and delta branches contiguously into memory starting from [`NodeId(0)`](NodeId).
    /// The node at `new_root` becomes `NodeId(0)` with `parent_edge = EdgeId::INVALID`, retaining all
    /// visited descendants, statistics, priors, and rewards. All dead branches outside the subtree are pruned.
    ///
    /// # Panics
    ///
    /// Panics if `new_root` is out of bounds or invalid.
    pub fn promote_subtree(&mut self, new_root: NodeId) {
        assert!(
            new_root.is_valid() && new_root.as_usize() < self.num_nodes(),
            "promote_subtree: new_root is invalid or out of bounds"
        );

        if new_root == NodeId(0) {
            self.nodes.parent_edge[0] = EdgeId::INVALID;
            return;
        }

        let old_num_nodes = self.num_nodes();
        let old_num_edges = self.num_edges();

        let mut old_to_new_node = vec![NodeId::INVALID; old_num_nodes];
        let mut old_to_new_edge = vec![EdgeId::INVALID; old_num_edges];

        let mut node_queue = std::collections::VecDeque::new();
        let mut reachable_nodes = Vec::new();
        let mut kept_edges = Vec::new();

        node_queue.push_back(new_root);
        old_to_new_node[new_root.as_usize()] = NodeId(0);
        reachable_nodes.push(new_root);

        while let Some(u) = node_queue.pop_front() {
            if self.nodes.status[u.as_usize()] == NodeStatus::Expanded {
                let first_e = self.nodes.first_child_edge[u.as_usize()];
                let count = self.nodes.num_children[u.as_usize()];
                for i in 0..count {
                    let e = EdgeId(first_e.0 + i);
                    let new_edge = EdgeId(kept_edges.len() as u32);
                    old_to_new_edge[e.as_usize()] = new_edge;
                    kept_edges.push(e.as_usize());

                    let primary_v = self.edges.child_node[e.as_usize()];
                    if primary_v.is_valid()
                        && old_to_new_node[primary_v.as_usize()] == NodeId::INVALID
                    {
                        let new_node = NodeId(reachable_nodes.len() as u32);
                        old_to_new_node[primary_v.as_usize()] = new_node;
                        reachable_nodes.push(primary_v);
                        node_queue.push_back(primary_v);
                    }

                    let mut b_idx = self.edges.first_branch[e.as_usize()];
                    while b_idx != u32::MAX {
                        let idx = b_idx as usize;
                        let v = self.branches.child_node[idx];
                        if v.is_valid() && old_to_new_node[v.as_usize()] == NodeId::INVALID {
                            let new_node = NodeId(reachable_nodes.len() as u32);
                            old_to_new_node[v.as_usize()] = new_node;
                            reachable_nodes.push(v);
                            node_queue.push_back(v);
                        }
                        b_idx = self.branches.next_branch[idx];
                    }
                }
            }
        }

        let new_num_nodes = reachable_nodes.len();
        let node_cap = self.nodes.parent_edge.capacity().max(new_num_nodes);
        let mut new_parent_edge = Vec::with_capacity(node_cap);
        let mut new_first_child_edge = Vec::with_capacity(node_cap);
        let mut new_num_children = Vec::with_capacity(node_cap);
        let mut new_agent = Vec::with_capacity(node_cap);
        let mut new_status = Vec::with_capacity(node_cap);

        for &old_u in &reachable_nodes {
            if old_u == new_root {
                new_parent_edge.push(EdgeId::INVALID);
            } else {
                let old_parent_e = self.nodes.parent_edge[old_u.as_usize()];
                new_parent_edge.push(old_to_new_edge[old_parent_e.as_usize()]);
            }

            let old_first_e = self.nodes.first_child_edge[old_u.as_usize()];
            if old_first_e.is_valid() {
                new_first_child_edge.push(old_to_new_edge[old_first_e.as_usize()]);
            } else {
                new_first_child_edge.push(EdgeId::INVALID);
            }

            new_num_children.push(self.nodes.num_children[old_u.as_usize()]);
            new_agent.push(self.nodes.agent[old_u.as_usize()]);
            new_status.push(self.nodes.status[old_u.as_usize()]);
        }

        let new_num_edges = kept_edges.len();
        let edge_cap = if std::mem::size_of::<Action>() == 0 {
            new_num_edges
        } else {
            self.edges.action.capacity().max(new_num_edges)
        };
        let mut new_action = Vec::with_capacity(edge_cap);
        let mut new_child_node = Vec::with_capacity(edge_cap);
        let mut new_first_branch = Vec::with_capacity(edge_cap);
        let mut new_reward = Vec::with_capacity(edge_cap);

        let branch_cap = self.branches.child_node.capacity();
        let mut new_branches_delta = if std::mem::size_of::<StepDelta>() == 0 {
            Vec::new()
        } else {
            Vec::with_capacity(branch_cap)
        };
        let mut new_branches_child = Vec::with_capacity(branch_cap);
        let mut new_branches_next = Vec::with_capacity(branch_cap);

        for &old_e in &kept_edges {
            new_action.push(self.edges.action[old_e].clone());
            let old_v = self.edges.child_node[old_e];
            let new_v = if old_v.is_valid() && old_to_new_node[old_v.as_usize()].is_valid() {
                old_to_new_node[old_v.as_usize()]
            } else {
                NodeId::INVALID
            };
            new_child_node.push(new_v);
            new_reward.push(self.edges.reward[old_e].clone());

            let mut old_b_idx = self.edges.first_branch[old_e];
            let mut new_head = u32::MAX;
            while old_b_idx != u32::MAX {
                let idx = old_b_idx as usize;
                let old_v = self.branches.child_node[idx];
                if old_v.is_valid() && old_to_new_node[old_v.as_usize()].is_valid() {
                    let branch_pos = new_branches_delta.len() as u32;
                    new_branches_delta.push(self.branches.delta[idx].clone());
                    new_branches_child.push(old_to_new_node[old_v.as_usize()]);
                    new_branches_next.push(new_head);
                    new_head = branch_pos;
                }
                old_b_idx = self.branches.next_branch[idx];
            }
            new_first_branch.push(new_head);
        }

        self.nodes = NodeArrays {
            parent_edge: new_parent_edge,
            first_child_edge: new_first_child_edge,
            num_children: new_num_children,
            agent: new_agent,
            status: new_status,
        };

        self.edges = EdgeArrays {
            action: new_action,
            child_node: new_child_node,
            first_branch: new_first_branch,
            reward: new_reward,
        };

        self.branches = DeltaBranchArrays {
            delta: new_branches_delta,
            child_node: new_branches_child,
            next_branch: new_branches_next,
        };

        self.stats.retain_edges(&kept_edges);
    }
}
