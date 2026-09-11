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
        Stats: EdgeStatsStore,
        D: AgentDynamics<Action = Action, Reward = Reward>,
        D::State: Clone,
        M: Model<D::State>,
        S: SelectionPolicy<Action, Reward, Stats>,
        B: BackupPolicy<Action, Reward, Stats, Evaluation>,
    {
        // 1. Root Initialization
        let root_actions = dynamics.actions(root_state);
        if root_actions.is_empty() {
            tree.mark_terminal(root);
            return;
        }

        if tree.node_status(root) == NodeStatus::Unexpanded {
            let eval = model.evaluate(root_state);
            tree.expand_node(root, &root_actions);
            backup.init_root(tree, root, &eval);
        }

        // 2. Iteration Loop
        for _ in 0..num_iterations {
            let mut state = root_state.clone();
            let mut path = Vec::new();
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
                    path.push(PathElement {
                        node: current_node,
                        edge,
                    });

                    let action = tree.edge_action(edge);
                    let transition = dynamics.step(state.clone(), action);
                    state = transition.next_state;

                    let child = tree.edge_child(edge);
                    if !child.is_valid() {
                        // Newly discovered node
                        tree.set_edge_reward(edge, transition.reward);
                        let inserted = tree.insert_node(edge, mcts_traits::AgentId(0));
                        current_node = inserted;
                        if transition.terminated {
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

            // Evaluation & Backup
            match tree.node_status(current_node) {
                NodeStatus::Terminal => {
                    backup.backup(tree, &path, None);
                }
                NodeStatus::Unexpanded => {
                    let actions = dynamics.actions(&state);
                    if actions.is_empty() {
                        tree.mark_terminal(current_node);
                        backup.backup(tree, &path, None);
                    } else {
                        let eval = model.evaluate(&state);
                        tree.expand_node(current_node, &actions);
                        backup.backup(tree, &path, Some(&eval));
                    }
                }
                NodeStatus::Expanded => {}
            }
        }
    }
}

