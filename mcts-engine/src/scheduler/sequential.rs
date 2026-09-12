use crate::backup::BackupPolicy;
use crate::search::descend_trajectory;
use crate::selection::SelectionPolicy;
use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, TreeStore};
use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Standard sequential Monte Carlo Tree Search (MCTS) scheduler.
///
/// Sweeps 1 simulation trajectory at a time sequentially from the root, expanding
/// leaves and updating node and edge values before beginning the next iteration.
pub struct SequentialScheduler;

impl SequentialScheduler {
    /// Executes `num_iterations` search passes starting from `root_state`.
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
        Stats: EdgeStatsStore,
        StepDelta: PartialEq + Clone,
        D: AgentDynamics<Action = Action, Reward = Reward, StepDelta = StepDelta>,
        D::State: Clone,
        M: Model<D::State>,
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

        let mut path = Vec::new();

        // 2. Iteration Loop
        for _ in 0..num_iterations {
            path.clear();
            let outcome =
                descend_trajectory(tree, dynamics, selection, root, root_state, &mut path, 0.0);
            let current_node = outcome.leaf_node;
            let state = outcome.leaf_state;

            // Evaluation & Backup
            match tree.node_status(current_node) {
                NodeStatus::Terminal => {
                    backup.backup(tree, &path, None);
                }
                NodeStatus::Unexpanded => {
                    dynamics.actions(&state, &mut scratch_actions);
                    if scratch_actions.is_empty() {
                        tree.mark_terminal(current_node);
                        backup.backup(tree, &path, None);
                    } else {
                        let eval = model.evaluate(&state);
                        tree.expand_node(current_node, &scratch_actions);
                        backup.backup(tree, &path, Some(&eval));
                    }
                }
                NodeStatus::Expanded => {}
            }
        }
    }
}
