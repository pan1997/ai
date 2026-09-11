use mcts_traits::AgentId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const INVALID: Self = Self(u32::MAX);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

impl EdgeId {
    pub const INVALID: Self = Self(u32::MAX);

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeStatus {
    #[default]
    Unexpanded,
    Expanded,
    Terminal,
}

pub trait EdgeStatsStore {
    fn resize(&mut self, new_len: usize);
    fn clear(&mut self);
}

pub trait PriorStore {
    fn set_prior(&mut self, edge: EdgeId, prior: f32);
}

pub trait VirtualLossStore {
    fn add_virtual_loss(&mut self, edge: EdgeId, weight: f32);
    fn remove_virtual_loss(&mut self, edge: EdgeId, weight: f32);
}

/// Structure-of-Arrays (SoA) memory layout for high cache locality and zero-allocation search traversals.
pub struct TreeStore<Action, Reward, Stats: EdgeStatsStore> {
    nodes: NodeArrays,
    edges: EdgeArrays<Action, Reward>,
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
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.parent_edge.len()
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.action.len()
    }

    #[inline]
    pub fn node_status(&self, node: NodeId) -> NodeStatus {
        self.nodes.status[node.as_usize()]
    }

    #[inline]
    pub fn is_expanded(&self, node: NodeId) -> bool {
        self.nodes.status[node.as_usize()] == NodeStatus::Expanded
    }

    #[inline]
    pub fn first_child_edge(&self, node: NodeId) -> EdgeId {
        self.nodes.first_child_edge[node.as_usize()]
    }

    #[inline]
    pub fn num_children(&self, node: NodeId) -> u32 {
        self.nodes.num_children[node.as_usize()]
    }

    #[inline]
    pub fn parent_edge(&self, node: NodeId) -> EdgeId {
        self.nodes.parent_edge[node.as_usize()]
    }

    #[inline]
    pub fn node_agent(&self, node: NodeId) -> AgentId {
        self.nodes.agent[node.as_usize()]
    }

    #[inline]
    pub fn edge_child(&self, edge: EdgeId) -> NodeId {
        self.edges.child_node[edge.as_usize()]
    }

    #[inline]
    pub fn edge_action(&self, edge: EdgeId) -> &Action {
        &self.edges.action[edge.as_usize()]
    }

    #[inline]
    pub fn edge_reward(&self, edge: EdgeId) -> Option<&Reward> {
        self.edges.reward[edge.as_usize()].as_ref()
    }

    #[inline]
    pub fn child_edges(&self, node: NodeId) -> impl Iterator<Item = EdgeId> {
        let first = self.first_child_edge(node);
        let count = self.num_children(node);
        (0..count).map(move |i| EdgeId(first.0 + i))
    }
}

impl<Action: Clone, Reward: Clone, Stats: EdgeStatsStore> TreeStore<Action, Reward, Stats> {
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

    pub fn insert_root(&mut self, agent: AgentId) -> NodeId {
        let node_id = NodeId(self.nodes.parent_edge.len() as u32);
        self.nodes.parent_edge.push(EdgeId::INVALID);
        self.nodes.first_child_edge.push(EdgeId::INVALID);
        self.nodes.num_children.push(0);
        self.nodes.agent.push(agent);
        self.nodes.status.push(NodeStatus::Unexpanded);
        node_id
    }

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

    pub fn mark_terminal(&mut self, node: NodeId) {
        let status = self.nodes.status[node.as_usize()];
        assert!(
            status == NodeStatus::Unexpanded || status == NodeStatus::Terminal,
            "mark_terminal: node must be Unexpanded or Terminal"
        );
        self.nodes.status[node.as_usize()] = NodeStatus::Terminal;
    }

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

