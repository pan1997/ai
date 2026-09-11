use crate::backup::{BackupPolicy, PathElement};
use crate::selection::SelectionPolicy;
use crate::tree_store::{
    EdgeStatsStore, NodeId, NodeStatus, TreeStore, VirtualLossStore,
};
use mcts_traits::{AgentDynamics, BatchedModel, Evaluation};

/// Batched MCTS scheduler for accelerating neural network evaluation within a single tree.
///
/// In each batch iteration:
/// 1. Simulates `batch_size` trajectories from the root using virtual loss to encourage diversity.
/// 2. Deduplicates unique leaf states to minimize redundant inference calls.
/// 3. Queries [`BatchedModel::evaluate_batch`] in a single GPU/tensor-friendly pass.
/// 4. Expands new leaves and backpropagates returns along all paths.
/// 5. Removes all temporary virtual losses.
pub struct BatchedScheduler {
    /// Number of concurrent simulation trajectories launched per batch.
    pub batch_size: usize,
    /// Penalty weight applied to traversed edges during selection to enforce path diversity.
    pub virtual_loss_weight: f32,
}

impl BatchedScheduler {
    /// Creates a new `BatchedScheduler` with the specified batch size and virtual loss weight.
    pub fn new(batch_size: usize, virtual_loss_weight: f32) -> Self {
        Self {
            batch_size,
            virtual_loss_weight,
        }
    }

    /// Executes `num_iterations` batched search passes (total simulations = `num_iterations * batch_size`).
    #[allow(clippy::too_many_arguments)]
    pub fn search<D, M, S, B, Action, Reward, Stats>(
        &self,
        tree: &mut TreeStore<Action, Reward, Stats>,
        dynamics: &D,
        model: &M,
        selection: &S,
        backup: &B,
        root: NodeId,
        root_state: &D::State,
        num_iterations: usize,
    ) where
        Action: Clone,
        Reward: Clone,
        Stats: EdgeStatsStore + VirtualLossStore,
        D: AgentDynamics<Action = Action, Reward = Reward>,
        D::State: Clone + PartialEq,
        M: BatchedModel<D::State>,
        S: SelectionPolicy<Action, Reward, Stats>,
        B: BackupPolicy<Action, Reward, Stats, Evaluation>,
    {
        // 1. Root Initialization
        let mut scratch_actions = Vec::new();
        dynamics.actions(root_state, &mut scratch_actions);
        if scratch_actions.is_empty() {
            tree.mark_terminal(root);
            return;
        }

        if tree.node_status(root) == NodeStatus::Unexpanded {
            let eval = model.evaluate(root_state);
            tree.expand_node(root, &scratch_actions);
            backup.init_root(tree, root, &eval);
        }

        // 2. Batched Iteration Loop
        for _ in 0..num_iterations {
            let mut paths: Vec<Vec<PathElement>> = Vec::with_capacity(self.batch_size);
            let mut leaf_nodes: Vec<NodeId> = Vec::with_capacity(self.batch_size);
            let mut leaf_states: Vec<D::State> = Vec::with_capacity(self.batch_size);

            for _ in 0..self.batch_size {
                let mut state = root_state.clone();
                let mut path = Vec::new();
                let mut current_node = root;

                while tree.node_status(current_node) == NodeStatus::Expanded {
                    if let Some(edge) = selection.select_child(tree, current_node) {
                        let first = tree.first_child_edge(current_node);
                        let count = tree.num_children(current_node);
                        assert!(
                            edge.0 >= first.0 && edge.0 < first.0 + count,
                            "SelectionPolicy: returned edge {} is not a valid child of node {}",
                            edge.0,
                            current_node.as_usize()
                        );
                        path.push(PathElement {
                            node: current_node,
                            edge,
                        });

                        // Apply virtual loss
                        tree.stats
                            .add_virtual_loss(edge, self.virtual_loss_weight);

                        let action = tree.edge_action(edge);
                        let outcome = dynamics.step(&mut state, action);

                        let child = tree.edge_child(edge);
                        if !child.is_valid() {
                            tree.set_edge_reward(edge, outcome.reward);
                            let inserted = tree.insert_node(edge, mcts_traits::AgentId(0));
                            current_node = inserted;
                            if outcome.terminated {
                                tree.mark_terminal(current_node);
                            }
                            break;
                        } else {
                            current_node = child;
                        }
                    } else {
                        break;
                    }
                }

                paths.push(path);
                leaf_nodes.push(current_node);
                leaf_states.push(state);
            }

            // Deduplicate unique leaves for evaluation
            let mut states_to_evaluate: Vec<D::State> = Vec::new();
            let mut unique_node_ids: Vec<NodeId> = Vec::new();
            let mut path_leaf_map: Vec<Option<usize>> = vec![None; self.batch_size];

            for b in 0..self.batch_size {
                let leaf_node = leaf_nodes[b];
                let state = &leaf_states[b];

                if tree.node_status(leaf_node) == NodeStatus::Terminal {
                    path_leaf_map[b] = None;
                } else if tree.node_status(leaf_node) == NodeStatus::Unexpanded {
                    dynamics.actions(state, &mut scratch_actions);
                    if scratch_actions.is_empty() {
                        tree.mark_terminal(leaf_node);
                        path_leaf_map[b] = None;
                    } else if let Some(pos) = unique_node_ids.iter().position(|&nid| nid == leaf_node) {
                        path_leaf_map[b] = Some(pos);
                    } else {
                        let pos = states_to_evaluate.len();
                        states_to_evaluate.push(state.clone());
                        unique_node_ids.push(leaf_node);
                        path_leaf_map[b] = Some(pos);
                    }
                } else {
                    path_leaf_map[b] = None;
                }
            }

            // Batched Evaluation
            let evals: Vec<Evaluation> = if !states_to_evaluate.is_empty() {
                let refs: Vec<&D::State> = states_to_evaluate.iter().collect();
                model.evaluate_batch(&refs)
            } else {
                Vec::new()
            };

            // Expand unique unexpanded leaves
            for (i, &leaf_node) in unique_node_ids.iter().enumerate() {
                let state = &states_to_evaluate[i];
                dynamics.actions(state, &mut scratch_actions);
                tree.expand_node(leaf_node, &scratch_actions);
            }

            // Backpropagate all paths and clean up virtual loss
            for b in 0..self.batch_size {
                let path = &paths[b];
                let eval_opt = path_leaf_map[b].map(|idx| &evals[idx]);

                backup.backup(tree, path, eval_opt);

                for pe in path {
                    tree.stats
                        .remove_virtual_loss(pe.edge, self.virtual_loss_weight);
                }
            }
        }
    }
}

