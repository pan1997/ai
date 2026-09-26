//! Single-Tree Information Set Monte Carlo Tree Search (Single-Tree ISMCTS) scheduler.

use crate::backup::{BackupPolicy, PathElement};
use crate::selection::ismcts::{IsmctsSelection, IsmctsStats};
use crate::tree_store::{NodeId, NodeStatus, TreeStore};
use mcts_traits::belief::BeliefSampler;
use mcts_traits::{ActionModel, AgentDynamics, Evaluation};

/// Single-Tree Information Set Monte Carlo Tree Search (Single-Tree ISMCTS) scheduler.
///
/// In contrast to Multi-Tree Determinization (which splits the iteration budget across multiple
/// independent trees), Single-Tree ISMCTS pools all simulation iterations into a single unified
/// search tree.
///
/// On each iteration pass:
/// 1. A fresh ground-truth state determinization $s^{(t)} \sim \mathbb{P}(S \mid \mathcal{O})$ is sampled.
/// 2. During tree traversal, candidate edges at each visited node are filtered to those legally
///    compatible with $s^{(t)}$.
/// 3. Availability counts $N_{\text{avail}}(s, a)$ are incremented for all compatible legal actions at visited nodes.
/// 4. Traversed visit counts $N(s, a)$ and value estimates are updated along the chosen trajectory path.
pub struct IsmctsScheduler;

impl IsmctsScheduler {
    /// Executes `num_iterations` Single-Tree ISMCTS search passes starting from `root`.
    #[allow(clippy::too_many_arguments)]
    pub fn search<D, M, B, Smp, Action, Reward, StepDelta, const N: usize>(
        &self,
        tree: &mut TreeStore<Action, Reward, IsmctsStats<N>, StepDelta>,
        dynamics: &D,
        model: &M,
        selection: &IsmctsSelection<N>,
        backup: &B,
        sampler: &mut Smp,
        context: &Smp::Context,
        root: NodeId,
        num_iterations: usize,
    ) where
        Action: Clone + PartialEq,
        Reward: Clone,
        StepDelta: PartialEq + Clone,
        D: AgentDynamics<Action = Action, Reward = Reward, StepDelta = StepDelta>,
        M: ActionModel<D::State, Action>,
        B: BackupPolicy<Action, Reward, IsmctsStats<N>, Evaluation, StepDelta>,
        Smp: BeliefSampler<State = D::State>,
    {
        // 1. Root Initialization
        let mut scratch_expand = Vec::new();
        let mut scratch_legal = Vec::new();

        if tree.node_status(root) == NodeStatus::Unexpanded {
            let initial_state = sampler.sample(context);
            dynamics.expand_actions(&initial_state, &mut scratch_expand);
            if scratch_expand.is_empty() {
                tree.mark_terminal(root);
                return;
            }

            let eval = model.evaluate_actions(&initial_state, &scratch_expand);
            tree.expand_node(root, &scratch_expand);
            backup.init_root(tree, root, &eval);
        }

        let mut path: Vec<PathElement<Reward>> = Vec::new();

        // 2. Main Trajectory Iteration Loop
        for _ in 0..num_iterations {
            path.clear();
            let mut sim_state = sampler.sample(context);
            let mut current_node = root;

            // 2.1 Trajectory Selection Descent
            while tree.node_status(current_node) == NodeStatus::Expanded {
                scratch_legal.clear();
                dynamics.actions(&sim_state, &mut scratch_legal);
                if scratch_legal.is_empty() {
                    tree.mark_terminal(current_node);
                    break;
                }

                // Update availability counts for all child edges matching legal actions in sim_state
                let first = tree.first_child_edge(current_node);
                let count = tree.num_children(current_node);
                for i in 0..count {
                    let edge = mcts_engine_edge_id(first.0 + i);
                    let action = tree.edge_action(edge);
                    if scratch_legal.contains(action) {
                        tree.stats.inc_avail_visits(edge);
                    }
                }

                // Select compatible child maximizing availability-weighted PUCT score
                let selected =
                    selection.select_compatible_child(tree, current_node, |_edge, action| {
                        scratch_legal.contains(action)
                    });

                if let Some(edge) = selected {
                    let action = tree.edge_action(edge).clone();
                    let outcome = dynamics.step(&mut sim_state, &action);

                    if tree.edge_reward(edge).is_none() {
                        tree.set_edge_reward(edge, outcome.reward.clone());
                    }

                    let agent = dynamics.current_agent(&sim_state);
                    let (child, is_new) = tree.get_or_insert_child(edge, &outcome.delta, agent);

                    path.push(PathElement {
                        node: current_node,
                        edge,
                        next_node: child,
                        reward: outcome.reward,
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
                    // No existing child edge is compatible with the legal actions in sim_state
                    break;
                }
            }

            // 2.2 Expansion & Evaluation at Leaf
            match tree.node_status(current_node) {
                NodeStatus::Terminal => {
                    backup.backup(tree, &path, None);
                }
                NodeStatus::Unexpanded => {
                    scratch_expand.clear();
                    dynamics.expand_actions(&sim_state, &mut scratch_expand);
                    if scratch_expand.is_empty() {
                        tree.mark_terminal(current_node);
                        backup.backup(tree, &path, None);
                    } else {
                        let eval = model.evaluate_actions(&sim_state, &scratch_expand);
                        tree.expand_node(current_node, &scratch_expand);
                        backup.backup(tree, &path, Some(&eval));
                    }
                }
                NodeStatus::Expanded => {
                    // Reached an expanded node with no compatible edge in this determinization.
                    // Evaluate leaf simulation state rather than injecting false 0.0 values.
                    let eval = model.evaluate(&sim_state);
                    backup.backup(tree, &path, Some(&eval));
                }
            }
        }
    }
}

#[inline(always)]
fn mcts_engine_edge_id(id: u32) -> crate::tree_store::EdgeId {
    crate::tree_store::EdgeId(id)
}
