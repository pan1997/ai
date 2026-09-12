//! Core tree descent algorithms for MCTS schedulers.
//!
//! Provides canonical trajectory descent logic shared across sequential, batched, and parallel schedulers.

use crate::backup::PathElement;
use crate::selection::SelectionPolicy;
use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, TreeStore};
use mcts_traits::AgentDynamics;

/// The outcome of descending a single search trajectory from a root or subtree node to a leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryOutcome<State> {
    /// The leaf node reached at the end of selection traversal.
    pub leaf_node: NodeId,
    /// The environment state at the leaf node.
    pub leaf_state: State,
}

/// Traverses a tree from `root` down to a leaf node using the provided selection policy.
///
/// At each expanded node along the trajectory:
/// 1. Queries `selection.select_child(tree, current_node)` to select an outgoing edge.
/// 2. If `virtual_loss_weight > 0.0`, applies virtual loss to the selected edge.
/// 3. Transitions `state` forward via `dynamics.step(&mut state, action)`.
/// 4. Stores the immediate step reward on the edge if not already recorded.
/// 5. Resolves or inserts the child node via `tree.get_or_insert_child(edge, &outcome.delta, agent)`.
/// 6. Appends a [`PathElement`] tracking `(current_node, edge, child_node)` to `path`.
/// 7. Terminates descent if the child node is newly inserted, or if either the transition
///    outcome or the child node is marked as terminal.
///
/// Traversal halts and returns the final [`TrajectoryOutcome`] when an unexpanded or terminal
/// node is encountered, or if the selection policy yields no legal action edge.
pub fn descend_trajectory<D, S, Action, Reward, Stats, StepDelta>(
    tree: &mut TreeStore<Action, Reward, Stats, StepDelta>,
    dynamics: &D,
    selection: &S,
    root: NodeId,
    root_state: &D::State,
    path: &mut Vec<PathElement>,
    virtual_loss_weight: f32,
) -> TrajectoryOutcome<D::State>
where
    Action: Clone,
    Reward: Clone,
    Stats: EdgeStatsStore,
    StepDelta: PartialEq + Clone,
    D: AgentDynamics<Action = Action, Reward = Reward, StepDelta = StepDelta>,
    D::State: Clone,
    S: SelectionPolicy<Action, Reward, Stats, StepDelta>,
{
    let mut state = root_state.clone();
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

            if virtual_loss_weight > 0.0 {
                tree.stats.add_virtual_loss(edge, virtual_loss_weight);
            }

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
            } else if outcome.terminated || tree.node_status(current_node) == NodeStatus::Terminal {
                tree.mark_terminal(current_node);
                break;
            }
        } else {
            break;
        }
    }

    TrajectoryOutcome {
        leaf_node: current_node,
        leaf_state: state,
    }
}
