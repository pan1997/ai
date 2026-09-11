use crate::backup::{BackupPolicy, PathElement};
use crate::selection::SelectionPolicy;
use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, TreeStore};
use mcts_traits::{BatchedAgentDynamics, BatchedModel, Evaluation};

pub struct MultiGameScheduler {
    pub batch_size: usize,
}

impl MultiGameScheduler {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search<D, M, S, B, Action, Reward, Stats>(
        &self,
        trees: &mut [TreeStore<Action, Reward, Stats>],
        roots: &[NodeId],
        root_states: &[&D::State],
        dynamics: &D,
        model: &M,
        selection: &S,
        backup: &B,
        num_iterations: usize,
    ) where
        Action: Clone,
        Reward: Clone,
        Stats: EdgeStatsStore,
        D: BatchedAgentDynamics<Action = Action, Reward = Reward>,
        D::State: Clone + PartialEq,
        M: BatchedModel<D::State>,
        S: SelectionPolicy<Action, Reward, Stats>,
        B: BackupPolicy<Action, Reward, Stats, Evaluation>,
    {
        assert_eq!(trees.len(), self.batch_size);
        assert_eq!(roots.len(), self.batch_size);
        assert_eq!(root_states.len(), self.batch_size);

        // 1. Root Initialization
        let mut active = vec![true; self.batch_size];
        for b in 0..self.batch_size {
            let actions = dynamics.actions(root_states[b]);
            if actions.is_empty() {
                trees[b].mark_terminal(roots[b]);
                active[b] = false;
            }
        }

        let mut root_states_to_eval = Vec::new();
        let mut active_indices = Vec::new();
        for b in 0..self.batch_size {
            if active[b] && trees[b].node_status(roots[b]) == NodeStatus::Unexpanded {
                root_states_to_eval.push(root_states[b]);
                active_indices.push(b);
            }
        }

        if !root_states_to_eval.is_empty() {
            let evals = model.evaluate_batch(&root_states_to_eval);
            for (i, &b) in active_indices.iter().enumerate() {
                let legal_actions = dynamics.actions(root_states[b]);
                trees[b].expand_node(roots[b], &legal_actions);
                backup.init_root(&mut trees[b], roots[b], &evals[i]);
            }
        }

        let iteration_trees: Vec<usize> = (0..self.batch_size).filter(|&b| active[b]).collect();
        if iteration_trees.is_empty() {
            return;
        }

        // 2. Iteration Loop
        for _ in 0..num_iterations {
            let mut paths: Vec<Vec<PathElement>> = vec![Vec::new(); self.batch_size];
            let mut current_node = roots.to_vec();
            let mut current_state: Vec<D::State> =
                root_states.iter().map(|&s| s.clone()).collect();
            let mut is_traversing = vec![false; self.batch_size];
            for &b in &iteration_trees {
                is_traversing[b] = true;
            }

            while is_traversing.iter().any(|&t| t) {
                let mut step_tree_indices = Vec::new();
                let mut active_states = Vec::new();
                let mut selected_actions = Vec::new();
                let mut selected_edges = Vec::new();

                for b in 0..self.batch_size {
                    if is_traversing[b] {
                        let node = current_node[b];
                        if trees[b].node_status(node) == NodeStatus::Expanded {
                            if let Some(edge) = selection.select_child(&trees[b], node) {
                                let first = trees[b].first_child_edge(node);
                                let count = trees[b].num_children(node);
                                assert!(
                                    edge.0 >= first.0 && edge.0 < first.0 + count,
                                    "SelectionPolicy: returned edge out of bounds"
                                );
                                paths[b].push(PathElement { node, edge });
                                step_tree_indices.push(b);
                                active_states.push(current_state[b].clone());
                                selected_actions.push(trees[b].edge_action(edge).clone());
                                selected_edges.push(edge);
                            } else {
                                is_traversing[b] = false;
                            }
                        } else {
                            is_traversing[b] = false;
                        }
                    }
                }

                if step_tree_indices.is_empty() {
                    break;
                }

                let mut transitions = Vec::new();
                dynamics.step_batch(&active_states, &selected_actions, &mut transitions);
                assert_eq!(transitions.len(), step_tree_indices.len());

                for (i, &b) in step_tree_indices.iter().enumerate() {
                    let transition = &transitions[i];
                    let edge = selected_edges[i];
                    current_state[b] = transition.next_state.clone();

                    let child = trees[b].edge_child(edge);
                    if !child.is_valid() {
                        trees[b].set_edge_reward(edge, transition.reward.clone());
                        let inserted = trees[b].insert_node(edge, mcts_traits::AgentId(0));
                        current_node[b] = inserted;
                        if transition.terminated {
                            trees[b].mark_terminal(inserted);
                        }
                        is_traversing[b] = false;
                    } else {
                        current_node[b] = child;
                    }
                }
            }

            // Deduplicate unique states
            let mut unique_states: Vec<D::State> = Vec::new();
            let mut consumers_list: Vec<Vec<(usize, NodeId)>> = Vec::new();
            let mut terminal_paths: Vec<(usize, Vec<PathElement>)> = Vec::new();

            for &b in &iteration_trees {
                let leaf_node = current_node[b];
                let state = &current_state[b];

                if trees[b].node_status(leaf_node) == NodeStatus::Terminal {
                    terminal_paths.push((b, paths[b].clone()));
                } else if trees[b].node_status(leaf_node) == NodeStatus::Unexpanded {
                    let actions = dynamics.actions(state);
                    if actions.is_empty() {
                        trees[b].mark_terminal(leaf_node);
                        terminal_paths.push((b, paths[b].clone()));
                    } else if let Some(pos) = unique_states.iter().position(|s| s == state) {
                        consumers_list[pos].push((b, leaf_node));
                    } else {
                        unique_states.push(state.clone());
                        consumers_list.push(vec![(b, leaf_node)]);
                    }
                } else {
                    terminal_paths.push((b, paths[b].clone()));
                }
            }

            // Batched Evaluation
            let evals: Vec<Evaluation> = if !unique_states.is_empty() {
                let refs: Vec<&D::State> = unique_states.iter().collect();
                model.evaluate_batch(&refs)
            } else {
                Vec::new()
            };

            // Expand unique leaves and backup
            for (u, eval) in evals.iter().enumerate() {
                let state = &unique_states[u];
                let actions = dynamics.actions(state);

                for &(tree_idx, leaf_node) in &consumers_list[u] {
                    trees[tree_idx].expand_node(leaf_node, &actions);
                    let path = &paths[tree_idx];
                    backup.backup(&mut trees[tree_idx], path, Some(eval));
                }
            }

            for (tree_idx, path) in terminal_paths {
                backup.backup(&mut trees[tree_idx], &path, None);
            }
        }
    }
}

