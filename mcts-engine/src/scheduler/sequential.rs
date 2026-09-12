use crate::backup::{BackupPolicy, PathElement};
use crate::selection::SelectionPolicy;
use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, TreeStore};
use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Single-threaded sequential MCTS execution scheduler.
///
/// Runs classic 1-by-1 simulation passes. Each iteration carries out:
/// 1. **Selection**: Descends from the root node to a leaf via [`SelectionPolicy`].
/// 2. **Expansion**: Adds legal action child edges if the leaf node is unexpanded.
/// 3. **Evaluation**: Evaluates the leaf state priors and value using [`Model`].
/// 4. **Backup**: Propagates values backwards along the trajectory via [`BackupPolicy`].
pub struct SequentialScheduler;

impl SequentialScheduler {
    /// Executes `num_iterations` sequential MCTS sweeps starting from `root`.
    ///
    /// # Generic Parameters
    ///
    /// - `D`: Environment dynamics implementing [`AgentDynamics`].
    /// - `M`: Evaluation model implementing [`Model`].
    /// - `S`: Selection policy implementing [`SelectionPolicy`].
    /// - `B`: Backup strategy implementing [`BackupPolicy`].
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
            let mut state = root_state.clone();
            path.clear();
            let mut current_node = root;

            // Selection traversal
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

                    let action = tree.edge_action(edge);
                    let outcome = dynamics.step(&mut state, action);

                    if tree.edge_reward(edge).is_none() {
                        tree.set_edge_reward(edge, outcome.reward);
                    }

                    let agent = dynamics.current_agent(&state);
                    let (child, is_new) = tree.get_or_insert_child(edge, &outcome.delta, agent);

                    path.push(PathElement {
                        node: current_node,
                        edge,
                        next_node: child,
                    });

                    current_node = child;
                    if is_new {
                        if outcome.terminated {
                            tree.mark_terminal(current_node);
                        }
                        break;
                    } else if outcome.terminated
                        || tree.node_status(current_node) == NodeStatus::Terminal
                    {
                        tree.mark_terminal(current_node);
                        break;
                    }
                } else {
                    break;
                }
            }

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
