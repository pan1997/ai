use crate::backup::{BackupPolicy, PathElement};
use crate::search::descend_trajectory;
use crate::selection::SelectionPolicy;
use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, TreeStore, VirtualLossStore};
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

/// RAII scope that ensures virtual loss applied to tree edges is reliably reverted on drop.
struct VirtualLossScope<'a, Action, Reward, Stats: EdgeStatsStore + VirtualLossStore, StepDelta> {
    tree: &'a mut TreeStore<Action, Reward, Stats, StepDelta>,
    edges: Vec<crate::tree_store::EdgeId>,
    weight: f32,
    disarmed: bool,
}

impl<'a, Action, Reward, Stats: EdgeStatsStore + VirtualLossStore, StepDelta> Drop
    for VirtualLossScope<'a, Action, Reward, Stats, StepDelta>
{
    fn drop(&mut self) {
        if !self.disarmed && self.weight > 0.0 {
            for &edge in &self.edges {
                self.tree.stats.remove_virtual_loss(edge, self.weight);
            }
        }
    }
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
    pub fn search<D, M, S, B, Action, Reward, Stats, StepDelta>(
        &self,
        tree: &mut TreeStore<Action, Reward, Stats, StepDelta>,
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
        StepDelta: PartialEq + Clone,
        D: AgentDynamics<Action = Action, Reward = Reward, StepDelta = StepDelta>,
        D::State: Clone + PartialEq,
        M: BatchedModel<D::State>,
        S: SelectionPolicy<Action, Reward, Stats, StepDelta>,
        B: BackupPolicy<Action, Reward, Stats, Evaluation, StepDelta>,
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

        // 2. Preallocate reusable scratch buffers across all batch iterations
        let mut paths: Vec<Vec<PathElement>> = (0..self.batch_size)
            .map(|_| Vec::with_capacity(32))
            .collect();
        let mut leaf_nodes: Vec<NodeId> = Vec::with_capacity(self.batch_size);
        let mut leaf_states: Vec<D::State> = Vec::with_capacity(self.batch_size);
        let mut states_to_evaluate: Vec<D::State> = Vec::with_capacity(self.batch_size);
        let mut unique_node_ids: Vec<NodeId> = Vec::with_capacity(self.batch_size);
        let mut unique_actions: Vec<Vec<Action>> = Vec::with_capacity(self.batch_size);
        let mut path_leaf_map: Vec<Option<usize>> = vec![None; self.batch_size];

        let mut scope = VirtualLossScope {
            tree,
            edges: Vec::with_capacity(self.batch_size * 16),
            weight: self.virtual_loss_weight,
            disarmed: false,
        };

        // 3. Batched Iteration Loop
        for _ in 0..num_iterations {
            leaf_nodes.clear();
            leaf_states.clear();
            states_to_evaluate.clear();
            unique_node_ids.clear();
            unique_actions.clear();
            path_leaf_map.fill(None);

            for path in paths.iter_mut().take(self.batch_size) {
                path.clear();
                let outcome = descend_trajectory(
                    scope.tree,
                    dynamics,
                    selection,
                    root,
                    root_state,
                    path,
                    self.virtual_loss_weight,
                );
                if self.virtual_loss_weight > 0.0 {
                    for pe in path.iter() {
                        scope.edges.push(pe.edge);
                    }
                }
                leaf_nodes.push(outcome.leaf_node);
                leaf_states.push(outcome.leaf_state);
            }

            // Deduplicate unique leaves for evaluation
            for b in 0..self.batch_size {
                let leaf_node = leaf_nodes[b];
                let state = &leaf_states[b];

                if scope.tree.node_status(leaf_node) == NodeStatus::Terminal {
                    path_leaf_map[b] = None;
                } else if scope.tree.node_status(leaf_node) == NodeStatus::Unexpanded {
                    dynamics.actions(state, &mut scratch_actions);
                    if scratch_actions.is_empty() {
                        scope.tree.mark_terminal(leaf_node);
                        path_leaf_map[b] = None;
                    } else if let Some(pos) =
                        unique_node_ids.iter().position(|&nid| nid == leaf_node)
                    {
                        path_leaf_map[b] = Some(pos);
                    } else {
                        let pos = states_to_evaluate.len();
                        states_to_evaluate.push(state.clone());
                        unique_node_ids.push(leaf_node);
                        unique_actions.push(scratch_actions.clone());
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

            // Expand unique unexpanded leaves using cached legal actions (no duplicate generation)
            for (i, &leaf_node) in unique_node_ids.iter().enumerate() {
                scope.tree.expand_node(leaf_node, &unique_actions[i]);
            }

            // Backpropagate all paths
            for b in 0..self.batch_size {
                let path = &paths[b];
                let eval_opt = path_leaf_map[b].map(|idx| &evals[idx]);
                backup.backup(scope.tree, path, eval_opt);
            }

            // Revert virtual loss for this completed batch
            if self.virtual_loss_weight > 0.0 {
                for &edge in &scope.edges {
                    scope
                        .tree
                        .stats
                        .remove_virtual_loss(edge, self.virtual_loss_weight);
                }
                scope.edges.clear();
            }
        }

        scope.disarmed = true;
    }
}
