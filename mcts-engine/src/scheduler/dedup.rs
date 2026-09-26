//! Leaf deduplication and expansion scheduling utilities for batched searches.
//!
//! When running batched MCTS searches across parallel trajectories (within a single tree
//! in [`BatchedScheduler`](crate::scheduler::BatchedScheduler) or across multiple concurrent
//! games in [`MultiGameScheduler`](crate::scheduler::MultiGameScheduler)), multiple simulation paths
//! may reach identical game states or identical unexpanded nodes.
//!
//! [`LeafDeduplicator`] coordinates:
//! 1. State deduplication for batched neural network evaluation (`model.evaluate_batch`).
//! 2. Caching legal action edge arrays to avoid duplicate calls to [`AgentDynamics::actions`](mcts_traits::AgentDynamics::actions).
//! 3. Ensuring each unexpanded node handle in a given tree is expanded at most once.
//! 4. Index mapping from batch item paths to their evaluation outputs for backpropagation.

use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, TreeStore};
use mcts_traits::AgentDynamics;

/// Represents an unexpanded node queued for edge expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpansionRequest {
    /// Index of the tree owning the node (0 for single-tree schedulers).
    pub tree_idx: usize,
    /// Handle of the node to expand.
    pub node: NodeId,
    /// Index into `LeafDeduplicator::unique_actions` containing the cached legal actions.
    pub action_idx: usize,
}

/// Scratch buffer container that deduplicates states and coordinates node expansion for batched MCTS.
#[derive(Debug, Clone)]
pub struct LeafDeduplicator<State, Action> {
    /// Deduplicated unique simulation states queued for batched evaluation.
    pub unique_states: Vec<State>,
    /// Cached legal action vectors corresponding to `unique_states`.
    pub unique_actions: Vec<Vec<Action>>,
    /// Unexpanded nodes queued for expansion along with their tree and action indices.
    pub nodes_to_expand: Vec<ExpansionRequest>,
    /// Mapping from batch item index to its index in `unique_states`, or `None` if terminal.
    pub path_eval_map: Vec<Option<usize>>,
}

impl<State, Action> LeafDeduplicator<State, Action> {
    /// Creates a new `LeafDeduplicator` with scratch capacity sized for `batch_size`.
    #[must_use]
    pub fn with_capacity(batch_size: usize) -> Self {
        Self {
            unique_states: Vec::with_capacity(batch_size),
            unique_actions: Vec::with_capacity(batch_size),
            nodes_to_expand: Vec::with_capacity(batch_size),
            path_eval_map: vec![None; batch_size],
        }
    }

    /// Resets all scratch storage while retaining preallocated heap capacities.
    #[inline]
    pub fn clear(&mut self, batch_size: usize) {
        self.unique_states.clear();
        self.unique_actions.clear();
        self.nodes_to_expand.clear();
        if self.path_eval_map.len() < batch_size {
            self.path_eval_map.resize(batch_size, None);
        } else {
            self.path_eval_map.truncate(batch_size);
            self.path_eval_map.fill(None);
        }
    }

    /// Registers a leaf node reached by trajectory `batch_idx` in tree `tree_idx`.
    ///
    /// - If the node is [`NodeStatus::Terminal`], records `None` in `path_eval_map`.
    /// - If the node is [`NodeStatus::Unexpanded`], queries legal actions. If no legal actions
    ///   exist, marks the node terminal and records `None`. Otherwise, queues the state for evaluation
    ///   and queues the node for expansion if not already scheduled.
    /// - If the node is [`NodeStatus::Expanded`], queues the leaf state for evaluation and records
    ///   its evaluation index without queuing expansion.
    #[allow(clippy::too_many_arguments)]
    pub fn register_leaf<D, R, Stats, StepDelta>(
        &mut self,
        batch_idx: usize,
        tree_idx: usize,
        tree: &mut TreeStore<Action, R, Stats, StepDelta>,
        dynamics: &D,
        leaf_node: NodeId,
        state: &D::State,
        scratch_actions: &mut Vec<Action>,
    ) where
        D: AgentDynamics<State = State, Action = Action>,
        State: Clone + PartialEq,
        Action: Clone,
        R: Clone,
        StepDelta: Clone,
        Stats: EdgeStatsStore,
    {
        match tree.node_status(leaf_node) {
            NodeStatus::Terminal => {
                self.path_eval_map[batch_idx] = None;
            }
            NodeStatus::Unexpanded => {
                dynamics.actions(state, scratch_actions);
                if scratch_actions.is_empty() {
                    tree.mark_terminal(leaf_node);
                    self.path_eval_map[batch_idx] = None;
                } else {
                    let act_idx =
                        if let Some(pos) = self.unique_states.iter().position(|s| s == state) {
                            pos
                        } else {
                            let pos = self.unique_states.len();
                            self.unique_states.push(state.clone());
                            self.unique_actions.push(scratch_actions.clone());
                            pos
                        };
                    self.path_eval_map[batch_idx] = Some(act_idx);
                    if !self
                        .nodes_to_expand
                        .iter()
                        .any(|req| req.tree_idx == tree_idx && req.node == leaf_node)
                    {
                        self.nodes_to_expand.push(ExpansionRequest {
                            tree_idx,
                            node: leaf_node,
                            action_idx: act_idx,
                        });
                    }
                }
            }
            NodeStatus::Expanded => {
                let state_idx =
                    if let Some(pos) = self.unique_states.iter().position(|s| s == state) {
                        pos
                    } else {
                        let pos = self.unique_states.len();
                        self.unique_states.push(state.clone());
                        self.unique_actions.push(Vec::new());
                        pos
                    };
                self.path_eval_map[batch_idx] = Some(state_idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::MultiAgentPuctStats;
    use crate::tree_store::EdgeId;
    use mcts_traits::{AgentId, GraphEnv};

    #[test]
    fn test_deduplicator_state_transposition_and_collision() {
        let mut env = GraphEnv::<1>::new(0);
        env.set_actions(42, vec![100, 200]);
        env.set_agent(42, AgentId(0));

        let stats = MultiAgentPuctStats::<1>::new();
        let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
        let n0 = tree.insert_root(AgentId(0));
        let edge0 = tree.expand_node(n0, &[1, 2]);
        let edge1 = EdgeId(edge0.0 + 1);
        let (n1, _) = tree.get_or_insert_child(edge0, &(), AgentId(0));
        let (n2, _) = tree.get_or_insert_child(edge1, &(), AgentId(0)); // n2 is a distinct second node

        let mut dedup = LeafDeduplicator::with_capacity(4);
        let mut scratch_actions = Vec::new();

        // 1. Batch item 0 reaches node n1 with state 42.
        dedup.register_leaf(0, 0, &mut tree, &env, n1, &42, &mut scratch_actions);
        assert_eq!(dedup.unique_states.len(), 1);
        assert_eq!(dedup.unique_actions.len(), 1);
        assert_eq!(dedup.unique_actions[0], vec![100, 200]);
        assert_eq!(dedup.nodes_to_expand.len(), 1);
        assert_eq!(dedup.path_eval_map[0], Some(0));

        // 2. Batch item 1 reaches a DIFFERENT node n2 with the SAME state 42 (transposition).
        dedup.register_leaf(1, 0, &mut tree, &env, n2, &42, &mut scratch_actions);
        // unique_states must NOT duplicate
        assert_eq!(dedup.unique_states.len(), 1);
        assert_eq!(dedup.unique_actions.len(), 1);
        // But n2 must be queued for expansion
        assert_eq!(dedup.nodes_to_expand.len(), 2);
        assert_eq!(dedup.nodes_to_expand[1].node, n2);
        assert_eq!(dedup.path_eval_map[1], Some(0));

        // 3. Batch item 2 reaches the SAME node n1 with state 42 (parallel search collision).
        dedup.register_leaf(2, 0, &mut tree, &env, n1, &42, &mut scratch_actions);
        // n1 must NOT be queued again!
        assert_eq!(dedup.nodes_to_expand.len(), 2);
        assert_eq!(dedup.path_eval_map[2], Some(0));
    }

    #[test]
    fn test_deduplicator_terminal_and_expanded_nodes() {
        let mut env = GraphEnv::<1>::new(0);
        env.set_actions(10, vec![]); // State 10 has no legal actions -> Terminal
        env.set_actions(20, vec![1]);
        env.set_agent(10, AgentId(0));
        env.set_agent(20, AgentId(0));

        let stats = MultiAgentPuctStats::<1>::new();
        let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
        let n0 = tree.insert_root(AgentId(0));
        let edge0 = tree.expand_node(n0, &[1, 2]);
        let edge1 = EdgeId(edge0.0 + 1);
        let (n1, _) = tree.get_or_insert_child(edge0, &(), AgentId(0)); // Unexpanded
        let (n2, _) = tree.get_or_insert_child(edge1, &(), AgentId(0));
        tree.expand_node(n2, &[999]); // n2 is Expanded

        let mut dedup = LeafDeduplicator::with_capacity(4);
        let mut scratch_actions = Vec::new();

        // 1. Unexpanded node n1 with 0 legal actions -> marked terminal, path_eval_map is None
        dedup.register_leaf(0, 0, &mut tree, &env, n1, &10, &mut scratch_actions);
        assert_eq!(tree.node_status(n1), NodeStatus::Terminal);
        assert_eq!(dedup.path_eval_map[0], None);
        assert_eq!(dedup.nodes_to_expand.len(), 0);

        // 2. Already Expanded node n2 -> evaluated, but not queued for expansion
        dedup.register_leaf(1, 0, &mut tree, &env, n2, &20, &mut scratch_actions);
        assert_eq!(dedup.unique_states.len(), 1);
        assert_eq!(dedup.unique_states[0], 20);
        assert_eq!(dedup.path_eval_map[1], Some(0));
        assert_eq!(dedup.nodes_to_expand.len(), 0); // No expansion queued!
    }
}
