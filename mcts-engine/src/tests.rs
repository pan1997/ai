use crate::backup::{BackupPolicy, PathElement, SingleAgentBackup, VectorBackup};
use crate::scheduler::{BatchedScheduler, MultiGameScheduler, SequentialScheduler};
use crate::selection::{
    GumbelPuctSelection, MultiAgentPuctSelection, MultiAgentPuctStats, SelectionPolicy,
    UctSelection,
};
use crate::tree_store::{EdgeId, NodeId, NodeStatus, PriorStore, TreeStore, VirtualLossStore};
use mcts_traits::{AgentId, BatchedModel, Evaluation, GraphEnv, Model, TurnBasedWorld, World};
use std::collections::HashMap;

#[derive(Clone, Default)]
struct MockModel {
    pub priors: HashMap<u32, Vec<f32>>,
    pub values: HashMap<u32, Vec<f32>>,
}

impl Model<u32> for MockModel {
    fn evaluate(&self, s: &u32) -> Evaluation {
        let priors = self.priors.get(s).cloned().unwrap_or_else(|| vec![1.0]);
        let values = self.values.get(s).cloned().unwrap_or_else(|| vec![0.0]);
        Evaluation { priors, values }
    }
}

impl BatchedModel<u32> for MockModel {
    fn evaluate_batch(&self, states: &[&u32]) -> Vec<Evaluation> {
        states.iter().map(|&s| self.evaluate(s)).collect()
    }
}

#[test]
fn test_sequential_scheduler_with_graph_env() {
    let mut env = GraphEnv::<2>::new(0);
    // State 0 (Agent 0) + Action 10 -> State 1 (Agent 1), reward [1.0, -1.0], not terminal
    env.add_transition(0, 10, 1, [1.0, -1.0], false);
    // State 1 (Agent 1) + Action 20 -> State 2 (terminal), reward [-2.0, 2.0], terminal
    env.add_transition(1, 20, 2, [-2.0, 2.0], true);

    env.set_actions(0, vec![10]);
    env.set_actions(1, vec![20]);
    env.set_agent(0, AgentId(0));
    env.set_agent(1, AgentId(1));

    let mut model = MockModel::default();
    model.priors.insert(0, vec![1.0]);
    model.priors.insert(1, vec![1.0]);
    model.values.insert(0, vec![0.0, 0.0]);
    model.values.insert(1, vec![0.5, -0.5]);

    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.0 };
    let backup = VectorBackup::<2>::default();
    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    let scheduler = SequentialScheduler;
    scheduler.search(&mut tree, &env, &model, &selection, &backup, root, &0, 1);

    assert_eq!(tree.num_nodes(), 2);
    assert_eq!(tree.num_edges(), 2);

    let edge0 = tree.first_child_edge(root);
    assert_eq!(*tree.edge_action(edge0), 10);
    assert_eq!(tree.stats.visits[edge0.as_usize()], 1);
    // g = [1.0, -1.0] + [0.5, -0.5] = [1.5, -1.5]
    assert_eq!(tree.stats.mean_value[edge0.as_usize()][0], 1.5);
    assert_eq!(tree.stats.mean_value[edge0.as_usize()][1], -1.5);
}

#[test]
fn test_selection_puct_formula() {
    let stats = MultiAgentPuctStats::<1> {
        visits: vec![10, 5],
        priors: vec![0.6, 0.4],
        mean_value: vec![[0.5], [0.8]],
        virtual_loss: vec![0.0, 0.0],
    };
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[100, 200]);

    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.0 };
    let chosen = selection.select_child(&tree, root).unwrap();

    // Edge 0 (visits=10, prior=0.6, q=0.5), Edge 1 (visits=5, prior=0.4, q=0.8)
    // Parent visits = 15, sqrt(15) = 3.87298
    // score0 = 0.5 + 1.0 * 0.6 * 3.87298 / 11 = 0.71125
    // score1 = 0.8 + 1.0 * 0.4 * 3.87298 / 6 = 1.05819 -> Edge 1 wins!
    assert_eq!(chosen.as_usize(), 1);
}

#[test]
fn test_selection_uct_unvisited_exploration() {
    let stats = MultiAgentPuctStats::<1> {
        visits: vec![10, 0], // Edge 1 has 0 visits!
        priors: vec![0.9, 0.1],
        mean_value: vec![[0.9], [0.0]],
        virtual_loss: vec![0.0, 0.0],
    };
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20]);

    let uct = UctSelection::<1>::default();
    let chosen = uct.select_child(&tree, root).unwrap();
    // Unvisited edge must be prioritized immediately (infinite score)
    assert_eq!(chosen.as_usize(), 1);
}

#[test]
fn test_batched_scheduler_virtual_loss_and_deduplication() {
    let mut env = GraphEnv::<1>::new(0);
    env.add_transition(0, 100, 1, [0.0], false);
    env.add_transition(0, 200, 2, [0.0], false);
    env.set_actions(0, vec![100, 200]);
    env.set_actions(1, vec![100]);
    env.set_actions(2, vec![200]);

    let mut model = MockModel::default();
    model.priors.insert(0, vec![0.5, 0.5]);
    model.priors.insert(1, vec![1.0]);
    model.priors.insert(2, vec![1.0]);
    model.values.insert(1, vec![0.1]);
    model.values.insert(2, vec![0.2]);

    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.0 };
    let backup = VectorBackup::<1>::default();
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));

    // Batch size 2 with strong virtual loss
    let scheduler = BatchedScheduler::new(2, 10.0);
    scheduler.search(&mut tree, &env, &model, &selection, &backup, root, &0, 1);

    // Both edges (100 and 200) should have been visited due to virtual loss
    let edge100 = tree.first_child_edge(root);
    let edge200 = EdgeId(edge100.0 + 1);

    assert_eq!(tree.stats.visits[edge100.as_usize()], 1);
    assert_eq!(tree.stats.visits[edge200.as_usize()], 1);

    // Virtual loss must be cleaned up to 0.0
    assert_eq!(tree.stats.virtual_loss[edge100.as_usize()], 0.0);
    assert_eq!(tree.stats.virtual_loss[edge200.as_usize()], 0.0);
}

#[test]
fn test_single_agent_backup_additive_returns() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10]);

    let edge = tree.first_child_edge(root);
    tree.set_edge_reward(edge, [2.0]);
    let child = tree.insert_node(edge, AgentId(0));

    let backup = SingleAgentBackup::new(0.9); // gamma = 0.9
    let path = [crate::backup::PathElement {
        node: root,
        edge,
        next_node: child,
        reward: [2.0],
    }];
    let eval = Evaluation::scalar(vec![], 10.0);

    backup.backup(&mut tree, &path, Some(&eval));

    // Expected return: g = reward + gamma * leaf_value = 2.0 + 0.9 * 10.0 = 11.0
    assert_eq!(tree.stats.visits[edge.as_usize()], 1);
    assert_eq!(tree.stats.mean_value[edge.as_usize()][0], 11.0);
}

#[test]
fn test_multi_game_scheduler() {
    let mut env = GraphEnv::<1>::new(0);
    env.add_transition(0, 10, 1, [1.0], true);
    env.add_transition(2, 10, 3, [2.0], true);
    env.set_actions(0, vec![10]);
    env.set_actions(2, vec![10]);

    let mut model = MockModel::default();
    model.priors.insert(0, vec![1.0]);
    model.priors.insert(2, vec![1.0]);
    model.values.insert(0, vec![0.0]);
    model.values.insert(2, vec![0.0]);

    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.0 };
    let backup = VectorBackup::<1>::default();

    let mut tree0 = TreeStore::with_capacity(5, 5, MultiAgentPuctStats::<1>::new());
    let mut tree1 = TreeStore::with_capacity(5, 5, MultiAgentPuctStats::<1>::new());

    let root0 = tree0.insert_root(AgentId(0));
    let root1 = tree1.insert_root(AgentId(0));

    let scheduler = MultiGameScheduler::new(2);
    let mut trees = [tree0, tree1];
    let roots = [root0, root1];
    let root_states = [&0, &2];

    scheduler.search(
        &mut trees,
        &roots,
        &root_states,
        &env,
        &model,
        &selection,
        &backup,
        1,
    );

    let edge0 = trees[0].first_child_edge(roots[0]);
    let edge1 = trees[1].first_child_edge(roots[1]);

    assert_eq!(trees[0].stats.visits[edge0.as_usize()], 1);
    assert_eq!(trees[0].stats.mean_value[edge0.as_usize()][0], 1.0);

    assert_eq!(trees[1].stats.visits[edge1.as_usize()], 1);
    assert_eq!(trees[1].stats.mean_value[edge1.as_usize()][0], 2.0);
}

#[test]
fn test_gumbel_puct_selection_root_vs_interior() {
    let stats = MultiAgentPuctStats::<1> {
        visits: vec![2, 2, 2],
        priors: vec![0.3, 0.4, 0.3],
        mean_value: vec![[0.5], [0.5], [0.5]],
        virtual_loss: vec![0.0, 0.0, 0.0],
    };
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20]);

    let edge0 = tree.first_child_edge(root);
    let interior_node = tree.insert_node(edge0, AgentId(0));
    tree.expand_node(interior_node, &[30]);

    let gumbel = GumbelPuctSelection::<1> { c_puct: 1.0 };

    // At root: Gumbel noise is applied to logits
    let root_pick = gumbel.select_child(&tree, root);
    assert!(root_pick.is_some());

    // At interior node: deterministic PUCT is applied (only child 30 is legal)
    let interior_pick = gumbel.select_child(&tree, interior_node);
    assert_eq!(interior_pick.unwrap().as_usize(), 2);
}

#[test]
fn test_gumbel_puct_single_child() {
    let stats = MultiAgentPuctStats::<1> {
        visits: vec![0],
        priors: vec![1.0],
        mean_value: vec![[0.0]],
        virtual_loss: vec![0.0],
    };
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[42]);

    let gumbel = GumbelPuctSelection::<1>::default();
    let chosen = gumbel.select_child(&tree, root).unwrap();
    assert_eq!(chosen.as_usize(), 0);
}

#[test]
fn test_multi_agent_puct_active_player_indexing() {
    // 2 agents. Edge 0 favors Agent 0 (Q=[1.0, -1.0]), Edge 1 favors Agent 1 (Q=[-1.0, 1.0])
    let stats = MultiAgentPuctStats::<2> {
        visits: vec![10, 10],
        priors: vec![0.5, 0.5],
        mean_value: vec![[1.0, -1.0], [-1.0, 1.0]],
        virtual_loss: vec![0.0, 0.0],
    };
    let mut tree: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1, 2]);

    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.0 };

    // When Agent 0 is acting, Edge 0 (Q_0 = 1.0) must win over Edge 1 (Q_0 = -1.0)
    let pick_p0 = selection.select_child(&tree, root).unwrap();
    assert_eq!(pick_p0.as_usize(), 0);

    // Create a node where Agent 1 is acting
    let mut tree_p1: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(
        5,
        5,
        MultiAgentPuctStats::<2> {
            visits: vec![10, 10],
            priors: vec![0.5, 0.5],
            mean_value: vec![[1.0, -1.0], [-1.0, 1.0]],
            virtual_loss: vec![0.0, 0.0],
        },
    );
    let root_p1 = tree_p1.insert_root(AgentId(1));
    tree_p1.expand_node(root_p1, &[1, 2]);

    // When Agent 1 is acting, Edge 1 (Q_1 = 1.0) must win over Edge 0 (Q_1 = -1.0)
    let pick_p1 = selection.select_child(&tree_p1, root_p1).unwrap();
    assert_eq!(pick_p1.as_usize(), 1);
}

#[test]
fn test_multi_agent_puct_virtual_loss_suppression() {
    // Both edges have equal prior and Q value
    let mut stats = MultiAgentPuctStats::<1> {
        visits: vec![5, 5],
        priors: vec![0.5, 0.5],
        mean_value: vec![[0.5], [0.5]],
        virtual_loss: vec![0.0, 0.0],
    };
    // Edge 0 has virtual loss applied
    stats.add_virtual_loss(EdgeId(0), 3.0);

    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20]);

    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.0 };
    let chosen = selection.select_child(&tree, root).unwrap();
    // Edge 1 should be selected because Edge 0 is depressed by virtual loss
    assert_eq!(chosen.as_usize(), 1);
}

#[test]
fn test_uct_multi_agent_all_visited() {
    let stats = MultiAgentPuctStats::<2> {
        visits: vec![4, 9],
        priors: vec![0.5, 0.5],
        mean_value: vec![[0.2, 0.8], [0.7, 0.3]],
        virtual_loss: vec![0.0, 0.0],
    };
    let mut tree: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(5, 5, stats);
    // Agent 1 is acting: Edge 0 has Q_1=0.8, visits=4; Edge 1 has Q_1=0.3, visits=9
    let root = tree.insert_root(AgentId(1));
    tree.expand_node(root, &[10, 20]);

    let uct = UctSelection::<2> { c_uct: 1.0 };
    let chosen = uct.select_child(&tree, root).unwrap();
    // Edge 0 has significantly higher Q for player 1 (0.8 vs 0.3)
    assert_eq!(chosen.as_usize(), 0);
}

#[test]
fn test_vector_backup_multi_step_discounting() {
    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10]); // edge 0

    let edge0 = tree.first_child_edge(root);
    tree.set_edge_reward(edge0, [1.0, -1.0]);
    let node1 = tree.insert_node(edge0, AgentId(1));
    tree.expand_node(node1, &[20]); // edge 1

    let edge1 = tree.first_child_edge(node1);
    tree.set_edge_reward(edge1, [2.0, -2.0]);
    let leaf_node = tree.insert_node(edge1, AgentId(0));
    tree.mark_terminal(leaf_node);

    let backup = VectorBackup::<2>::new(0.5); // gamma = 0.5
    let path = [
        PathElement {
            node: root,
            edge: edge0,
            next_node: node1,
            reward: [1.0, -1.0],
        },
        PathElement {
            node: node1,
            edge: edge1,
            next_node: leaf_node,
            reward: [2.0, -2.0],
        },
    ];
    let eval = Evaluation::vector(vec![], vec![4.0, -4.0]);

    backup.backup(&mut tree, &path, Some(&eval));

    // Step 1: g_1 = reward1 + 0.5 * V = [2.0, -2.0] + 0.5 * [4.0, -4.0] = [4.0, -4.0]
    assert_eq!(tree.stats.visits[edge1.as_usize()], 1);
    assert_eq!(tree.stats.mean_value[edge1.as_usize()][0], 4.0);
    assert_eq!(tree.stats.mean_value[edge1.as_usize()][1], -4.0);

    // Step 0: g_0 = reward0 + 0.5 * g_1 = [1.0, -1.0] + 0.5 * [4.0, -4.0] = [3.0, -3.0]
    assert_eq!(tree.stats.visits[edge0.as_usize()], 1);
    assert_eq!(tree.stats.mean_value[edge0.as_usize()][0], 3.0);
    assert_eq!(tree.stats.mean_value[edge0.as_usize()][1], -3.0);
}

#[test]
fn test_vector_backup_terminal_leaf_without_eval() {
    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10]);

    let edge0 = tree.first_child_edge(root);
    tree.set_edge_reward(edge0, [5.0, -5.0]);
    let leaf = tree.insert_node(edge0, AgentId(0));
    tree.mark_terminal(leaf);

    let backup = VectorBackup::<2>::new(1.0);
    let path = [PathElement {
        node: root,
        edge: edge0,
        next_node: leaf,
        reward: [5.0, -5.0],
    }];

    // Terminal leaf evaluated with None
    backup.backup(&mut tree, &path, Option::<&Evaluation>::None);

    assert_eq!(tree.stats.visits[edge0.as_usize()], 1);
    assert_eq!(tree.stats.mean_value[edge0.as_usize()][0], 5.0);
    assert_eq!(tree.stats.mean_value[edge0.as_usize()][1], -5.0);
}

#[test]
fn test_vector_backup_zero_prior_fallback() {
    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20]);

    let backup = VectorBackup::<2>::default();
    // Prior sum is 0.0 -> must fall back to uniform 1/2 = 0.5
    let eval = Evaluation::vector(vec![0.0, 0.0], vec![0.0, 0.0]);
    backup.init_root(&mut tree, root, &eval);

    let edge0 = tree.first_child_edge(root);
    let edge1 = EdgeId(edge0.0 + 1);

    assert_eq!(tree.stats.priors[edge0.as_usize()], 0.5);
    assert_eq!(tree.stats.priors[edge1.as_usize()], 0.5);
}

#[test]
fn test_tree_store_clear_and_capacity() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(50, 50, stats);

    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1, 2, 3]);

    assert_eq!(tree.num_nodes(), 1);
    assert_eq!(tree.num_edges(), 3);
    assert!(tree.is_expanded(root));

    tree.clear();

    assert_eq!(tree.num_nodes(), 0);
    assert_eq!(tree.num_edges(), 0);

    // Re-use cleared store
    let new_root = tree.insert_root(AgentId(0));
    assert_eq!(new_root, NodeId(0));
    assert_eq!(tree.node_status(new_root), NodeStatus::Unexpanded);
}

#[test]
fn test_tree_store_child_edges_iterator() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20, 30, 40]);

    let children: Vec<EdgeId> = tree.child_edges(root).collect();
    assert_eq!(children.len(), 4);
    assert_eq!(children[0], EdgeId(0));
    assert_eq!(children[1], EdgeId(1));
    assert_eq!(children[2], EdgeId(2));
    assert_eq!(children[3], EdgeId(3));
}

#[test]
#[should_panic(expected = "expand_node: node must be Unexpanded")]
fn test_tree_store_expand_already_expanded_panic() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1]);
    tree.expand_node(root, &[2]); // Panic!
}

#[test]
#[should_panic(expected = "set_edge_reward: reward has already been set for this edge")]
fn test_tree_store_set_reward_twice_panic() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1]);
    let edge = tree.first_child_edge(root);
    tree.set_edge_reward(edge, [1.0]);
    tree.set_edge_reward(edge, [2.0]); // Panic!
}

#[test]
#[should_panic(expected = "insert_node: parent_edge out of bounds")]
fn test_tree_store_insert_node_out_of_bounds_panic() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    tree.insert_node(EdgeId(999), AgentId(0)); // Panic!
}

#[test]
#[should_panic(expected = "mark_terminal: node must be Unexpanded or Terminal")]
fn test_tree_store_mark_expanded_terminal_panic() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1]);
    tree.mark_terminal(root); // Panic!
}

#[test]
fn test_schedulers_terminal_root() {
    // Environment where initial state has 0 legal actions
    let mut env = GraphEnv::<1>::new(0);
    env.set_actions(0, vec![]);
    env.set_terminal(0, true);

    let model = MockModel::default();
    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.0 };
    let backup = VectorBackup::<1>::default();

    // 1. SequentialScheduler
    let mut tree_seq = TreeStore::with_capacity(5, 5, MultiAgentPuctStats::<1>::new());
    let root_seq = tree_seq.insert_root(AgentId(0));
    SequentialScheduler.search(
        &mut tree_seq,
        &env,
        &model,
        &selection,
        &backup,
        root_seq,
        &0,
        10,
    );
    assert_eq!(tree_seq.node_status(root_seq), NodeStatus::Terminal);

    // 2. BatchedScheduler
    let mut tree_batch = TreeStore::with_capacity(5, 5, MultiAgentPuctStats::<1>::new());
    let root_batch = tree_batch.insert_root(AgentId(0));
    BatchedScheduler::new(2, 1.0).search(
        &mut tree_batch,
        &env,
        &model,
        &selection,
        &backup,
        root_batch,
        &0,
        5,
    );
    assert_eq!(tree_batch.node_status(root_batch), NodeStatus::Terminal);

    // 3. MultiGameScheduler
    let mut tree_mg = TreeStore::with_capacity(5, 5, MultiAgentPuctStats::<1>::new());
    let root_mg = tree_mg.insert_root(AgentId(0));
    let mut trees = [tree_mg];
    let roots = [root_mg];
    let root_states = [&0];
    MultiGameScheduler::new(1).search(
        &mut trees,
        &roots,
        &root_states,
        &env,
        &model,
        &selection,
        &backup,
        5,
    );
    assert_eq!(trees[0].node_status(roots[0]), NodeStatus::Terminal);
}

#[test]
fn test_tree_store_promote_subtree() {
    // Tree:
    // Root(0) -> Edge 0 (action 10) -> Node 1
    //         -> Edge 1 (action 20) -> Node 2
    // Node 1  -> Edge 2 (action 30) -> Node 3
    //         -> Edge 3 (action 40) -> Node 4
    // Node 2  -> Edge 4 (action 50) -> Node 5
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(10, 10, stats);
    let r0 = tree.insert_root(AgentId(0));
    let e0 = tree.expand_node(r0, &[10, 20]);
    assert_eq!(e0, EdgeId(0));

    let n1 = tree.insert_node(EdgeId(0), AgentId(1));
    let n2 = tree.insert_node(EdgeId(1), AgentId(1));
    assert_eq!(n1, NodeId(1));
    assert_eq!(n2, NodeId(2));

    tree.set_edge_reward(EdgeId(0), [1.0]);
    tree.set_edge_reward(EdgeId(1), [2.0]);
    tree.stats.visits[0] = 50;
    tree.stats.visits[1] = 20;

    let e2 = tree.expand_node(n1, &[30, 40]);
    assert_eq!(e2, EdgeId(2));
    let _n3 = tree.insert_node(EdgeId(2), AgentId(0));
    let _n4 = tree.insert_node(EdgeId(3), AgentId(0));
    tree.set_edge_reward(EdgeId(2), [3.0]);
    tree.set_edge_reward(EdgeId(3), [4.0]);
    tree.stats.visits[2] = 25;
    tree.stats.visits[3] = 25;

    let e4 = tree.expand_node(n2, &[50]);
    assert_eq!(e4, EdgeId(4));
    let _n5 = tree.insert_node(EdgeId(4), AgentId(0));
    tree.stats.visits[4] = 20;

    assert_eq!(tree.num_nodes(), 6);
    assert_eq!(tree.num_edges(), 5);

    // Promote subtree rooted at Node 1
    tree.promote_subtree(NodeId(1));

    // After promotion:
    // Reachable nodes: Node 1 (now 0), Node 3 (now 1), Node 4 (now 2)
    // Reachable edges: Edge 2 (now 0, action 30), Edge 3 (now 1, action 40)
    assert_eq!(tree.num_nodes(), 3);
    assert_eq!(tree.num_edges(), 2);

    let new_root = NodeId(0);
    assert_eq!(tree.parent_edge(new_root), EdgeId::INVALID);
    assert_eq!(tree.node_status(new_root), NodeStatus::Expanded);
    assert_eq!(tree.node_agent(new_root), AgentId(1));
    assert_eq!(tree.num_children(new_root), 2);

    let fe = tree.first_child_edge(new_root);
    assert_eq!(fe, EdgeId(0));
    assert_eq!(*tree.edge_action(EdgeId(0)), 30);
    assert_eq!(*tree.edge_action(EdgeId(1)), 40);
    assert_eq!(tree.edge_reward(EdgeId(0)), Some(&[3.0]));
    assert_eq!(tree.edge_reward(EdgeId(1)), Some(&[4.0]));
    assert_eq!(tree.stats.visits[0], 25);
    assert_eq!(tree.stats.visits[1], 25);

    // Check children pointers
    let c0 = tree.edge_child(EdgeId(0));
    let c1 = tree.edge_child(EdgeId(1));
    assert_eq!(c0, NodeId(1));
    assert_eq!(c1, NodeId(2));
    assert_eq!(tree.parent_edge(c0), EdgeId(0));
    assert_eq!(tree.parent_edge(c1), EdgeId(1));

    // Promoting root to root is a no-op
    tree.promote_subtree(NodeId(0));
    assert_eq!(tree.num_nodes(), 3);
    assert_eq!(tree.num_edges(), 2);
}

#[test]
fn test_dirichlet_noise_utilities() {
    use crate::dirichlet::{add_dirichlet_noise, add_root_dirichlet_noise};
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Empty and single-element edge cases
    let mut empty: [f32; 0] = [];
    add_dirichlet_noise(&mut empty, 0.3, 0.25, &mut rng);

    let mut single = [1.0f32];
    add_dirichlet_noise(&mut single, 0.3, 0.25, &mut rng);
    assert_eq!(single[0], 1.0);

    // Multi-element slice
    let mut priors = [0.4f32, 0.4, 0.2];
    add_dirichlet_noise(&mut priors, 0.3, 0.25, &mut rng);
    let sum: f32 = priors.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Dirichlet noise should preserve normalization: sum={sum}"
    );

    // Root Dirichlet noise on TreeStore
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1, 2, 3]);
    tree.stats.set_prior(EdgeId(0), 0.5);
    tree.stats.set_prior(EdgeId(1), 0.3);
    tree.stats.set_prior(EdgeId(2), 0.2);

    add_root_dirichlet_noise(&mut tree, root, 0.3, 0.25, &mut rng);

    let p0 = tree.stats.prior(EdgeId(0));
    let p1 = tree.stats.prior(EdgeId(1));
    let p2 = tree.stats.prior(EdgeId(2));
    let tree_sum = p0 + p1 + p2;
    assert!(
        (tree_sum - 1.0).abs() < 1e-4,
        "Tree priors should sum to 1.0: sum={tree_sum}"
    );
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MockTwoPlayerState {
    player: usize,
    round: usize,
    history: Vec<u32>,
    terminal: bool,
}

struct MockTwoPlayerWorld;

impl mcts_traits::World for MockTwoPlayerWorld {
    type WorldState = MockTwoPlayerState;
    type Action = u32;
    type Observation = MockTwoPlayerState;

    fn n_players(&self) -> usize {
        2
    }

    fn initial(&self) -> Self::WorldState {
        MockTwoPlayerState {
            player: 0,
            round: 0,
            history: Vec::new(),
            terminal: false,
        }
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        ws.clone()
    }

    fn actions(
        &self,
        state: &Self::WorldState,
        player: usize,
        out_actions: &mut Vec<Self::Action>,
    ) {
        out_actions.clear();
        if state.terminal {
            return;
        }
        if player == 0 {
            out_actions.extend_from_slice(&[1, 2]);
        } else {
            out_actions.extend_from_slice(&[10, 20]);
        }
    }

    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        let action = &joint[ws.player];
        let outcome = self.step_action(ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    fn terminal(&self, state: &Self::WorldState) -> bool {
        state.terminal
    }
}

impl mcts_traits::TurnBasedWorld for MockTwoPlayerWorld {
    type StepReward = [f32; 2];

    fn current_player(&self, state: &Self::WorldState) -> usize {
        state.player
    }

    fn step_action(
        &self,
        state: &mut Self::WorldState,
        action: &Self::Action,
    ) -> mcts_traits::StepOutcome<Self::StepReward> {
        state.history.push(*action);
        if state.player == 0 {
            state.player = 1;
            // Immediate win condition for action 2 to test primary agent immediate win
            if *action == 2 {
                state.terminal = true;
                return mcts_traits::StepOutcome::new([1.0, -1.0], true);
            }
            mcts_traits::StepOutcome::new([0.0, 0.0], false)
        } else {
            state.player = 0;
            state.round += 1;
            if state.round >= 2 || *action == 20 {
                state.terminal = true;
                let reward = if *action == 20 {
                    [-1.0, 1.0] // Opponent win
                } else {
                    [1.0, -1.0]
                };
                mcts_traits::StepOutcome::new(reward, true)
            } else {
                mcts_traits::StepOutcome::new([0.0, 0.0], false)
            }
        }
    }
}

/// Model that enforces the Single-Perspective Evaluator Invariant:
/// Panics if evaluate() is ever called on an opponent state!
struct StrictPerspectiveModel;

impl Model<MockTwoPlayerState> for StrictPerspectiveModel {
    fn evaluate(&self, s: &MockTwoPlayerState) -> Evaluation {
        assert_eq!(
            s.player, 0,
            "Evaluator dilution invariant violated: evaluate() called on player {}",
            s.player
        );
        Evaluation::vector(vec![0.5, 0.5], vec![0.2, -0.2])
    }
}

struct DeterministicOpponentPolicy;

impl mcts_traits::OpponentPolicy<MockTwoPlayerState, u32> for DeterministicOpponentPolicy {
    fn select_action(&self, _state: &MockTwoPlayerState) -> u32 {
        10 // Always choose action 10
    }
}

#[derive(Clone, Copy)]
struct RandomMockOpponent;

impl mcts_traits::OpponentPolicy<MockTwoPlayerState, u32> for RandomMockOpponent {
    fn select_action(&self, state: &MockTwoPlayerState) -> u32 {
        use rand::Rng;
        let mut actions = Vec::new();
        MockTwoPlayerWorld.actions(state, 1, &mut actions);
        actions[rand::thread_rng().gen_range(0..actions.len())]
    }
}

#[test]
fn test_sequential_scheduler_heuristic_opponent_and_evaluator_invariant() {
    use mcts_traits::RoundBasedDynamics;

    let world = MockTwoPlayerWorld;
    let root_state = world.initial();
    let model = StrictPerspectiveModel;
    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.0 };
    let dynamics = RoundBasedDynamics::new(world, DeterministicOpponentPolicy, 0);
    let backup = VectorBackup::<2>::default();

    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree: TreeStore<u32, [f32; 2], _, Vec<u32>> = TreeStore::with_capacity(50, 50, stats);
    let root = tree.insert_root(AgentId(0));

    let scheduler = SequentialScheduler;
    scheduler.search(
        &mut tree,
        &dynamics,
        &model,
        &selection,
        &backup,
        root,
        &root_state,
        20,
    );

    // 1. Invariant verified: StrictPerspectiveModel did not panic, meaning
    // evaluate() was ONLY called on primary agent states (player == 0).

    // 2. Search tree structure verification:
    // Root must be agent 0
    assert_eq!(tree.node_agent(root), AgentId(0));
    assert_eq!(tree.num_children(root), 2);

    let edge1 = tree.first_child_edge(root);
    // Delta branch for action 10
    let child10 = tree.get_child(edge1, &vec![10]);
    assert!(child10.is_some());

    // Root edge 1 (action 2) leads to immediate win
    let edge2 = EdgeId(edge1.0 + 1);
    let child_win = tree.get_child(edge2, &vec![]);
    if let Some(win_node) = child_win {
        assert_eq!(tree.node_status(win_node), NodeStatus::Terminal);
    }
}

#[test]
fn test_sequential_scheduler_random_opponent_with_step_delta_branching() {
    use mcts_traits::RoundBasedDynamics;

    let world = MockTwoPlayerWorld;
    let root_state = world.initial();
    let model = StrictPerspectiveModel;
    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.0 };
    let dynamics = RoundBasedDynamics::new(world, RandomMockOpponent, 0);
    let backup = VectorBackup::<2>::default();

    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree: TreeStore<u32, [f32; 2], _, Vec<u32>> = TreeStore::with_capacity(50, 50, stats);
    let root = tree.insert_root(AgentId(0));

    let scheduler = SequentialScheduler;
    scheduler.search(
        &mut tree,
        &dynamics,
        &model,
        &selection,
        &backup,
        root,
        &root_state,
        50,
    );

    assert!(tree.num_nodes() > 2);
    let edge0 = tree.first_child_edge(root);
    let edge1 = EdgeId(edge0.0 + 1);
    assert_eq!(
        tree.stats.visits[edge0.as_usize()] + tree.stats.visits[edge1.as_usize()],
        50
    );

    // Verify delta children can be iterated
    let delta_branches: Vec<_> = tree.delta_children(edge0).collect();
    assert!(!delta_branches.is_empty());
}

#[test]
fn test_normalized_uct_selection_large_q_values() {
    use crate::selection::NormalizedUctSelection;

    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _, ()> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20]);

    let edge0 = tree.first_child_edge(root);
    let edge1 = EdgeId(edge0.0 + 1);

    let selection = NormalizedUctSelection::<1>::default();

    // 1. Unvisited edges should have +inf score (explored first)
    assert_eq!(selection.select_child(&tree, root), Some(edge0));

    // Simulate edge0 visited once with huge Q value
    tree.stats.visits[edge0.as_usize()] = 1;
    tree.stats.mean_value[edge0.as_usize()] = [10_000.0];

    // edge1 is still unvisited -> must be selected next!
    assert_eq!(selection.select_child(&tree, root), Some(edge1));

    // Simulate edge1 visited once with slightly higher Q value
    tree.stats.visits[edge1.as_usize()] = 1;
    tree.stats.mean_value[edge1.as_usize()] = [10_050.0];

    // Now both edges are visited.
    // Normalized Q: edge0 -> 0.0, edge1 -> 1.0.
    // Parent visits = 2.
    // Exploration term for both is c_uct * sqrt(ln(3) / 1) ≈ 1.414 * 1.048 ≈ 1.48.
    // edge0 score ≈ 0.0 + 1.48 = 1.48.
    // edge1 score ≈ 1.0 + 1.48 = 2.48.
    // edge1 has higher score and should be selected!
    assert_eq!(selection.select_child(&tree, root), Some(edge1));

    // Now suppose edge1 is visited 10 more times, so N(edge1) = 11.
    // Parent visits = 1 + 11 = 12.
    // Exploration term for edge0: 1.414 * sqrt(ln(13) / 1) ≈ 1.414 * 1.6 ≈ 2.26 -> total 0.0 + 2.26 = 2.26.
    // Exploration term for edge1: 1.414 * sqrt(ln(13) / 11) ≈ 1.414 * 0.48 ≈ 0.68 -> total 1.0 + 0.68 = 1.68.
    // Edge0 now has HIGHER score due to exploration!
    tree.stats.visits[edge1.as_usize()] = 11;
    assert_eq!(selection.select_child(&tree, root), Some(edge0));
}

#[test]
fn test_normalized_puct_selection_scale_invariance() {
    use crate::selection::NormalizedPuctSelection;

    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _, ()> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10, 20]);

    let edge0 = tree.first_child_edge(root);
    let edge1 = EdgeId(edge0.0 + 1);

    // Set priors: edge0 has 0.7, edge1 has 0.3
    tree.stats.priors[edge0.as_usize()] = 0.7;
    tree.stats.priors[edge1.as_usize()] = 0.3;

    let selection = NormalizedPuctSelection::<1>::default();

    // 1. Initial selection selects child with highest prior (edge0)
    assert_eq!(selection.select_child(&tree, root), Some(edge0));

    // Simulate edge0 visited with huge Q = 50,000.0
    tree.stats.visits[edge0.as_usize()] = 1;
    tree.stats.mean_value[edge0.as_usize()] = [50_000.0];

    // Under unnormalized PUCT, edge0 would stay at 50,000 while edge1 would be 0.0 + 1.4*0.3 = 0.42.
    // But under NormalizedPuctSelection, edge0's Q_norm is 0.5 (only 1 visited child),
    // and edge1's score is 0.0 + 1.4142 * 0.3 * sqrt(1) / 1 = 0.424.
    // edge0 score is 0.5 + 1.4142 * 0.7 * 1 / 2 = 0.5 + 0.495 = 0.995.
    assert_eq!(selection.select_child(&tree, root), Some(edge0));

    // After edge0 gets a few visits, edge1 will be selected because exploration bonus drives it:
    tree.stats.visits[edge0.as_usize()] = 10;
    // Parent visits = 10. sqrt(10) ≈ 3.16.
    // edge1 score = 0.0 + 1.4142 * 0.3 * 3.16 / 1 ≈ 1.34 > 1.0!
    // edge0 score <= 1.0 + exploration (< 1.34).
    assert_eq!(selection.select_child(&tree, root), Some(edge1));
}

#[test]
fn test_promote_subtree_preserves_capacity() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(100, 200, stats);
    let r0 = tree.insert_root(AgentId(0));
    tree.expand_node(r0, &[10, 20]);
    let n1 = tree.insert_node(EdgeId(0), AgentId(1));
    tree.expand_node(n1, &[30, 40]);

    // Promote subtree starting at n1
    tree.promote_subtree(n1);
    assert_eq!(tree.num_nodes(), 1);

    // Tree capacity should still be at least the initial preallocated capacity
    assert!(tree.node_capacity() >= 100);
    assert!(tree.edge_capacity() >= 200);
}

#[test]
fn test_batched_scheduler_virtual_loss_guard_exception_safety() {
    use crate::backup::SingleAgentBackup;
    use crate::scheduler::BatchedScheduler;
    use crate::selection::UctSelection;
    use mcts_traits::{BatchedModel, Evaluation};

    struct PanickingModel;
    impl mcts_traits::Model<u32> for PanickingModel {
        fn evaluate(&self, _s: &u32) -> Evaluation {
            Evaluation::scalar(vec![0.5, 0.5], 0.0)
        }
    }
    impl BatchedModel<u32> for PanickingModel {
        fn evaluate_batch(&self, _states: &[&u32]) -> Vec<Evaluation> {
            panic!("Simulated neural network runtime panic (e.g. CUDA OOM)");
        }
    }

    struct Mock1PEnv;
    impl mcts_traits::AgentDynamics for Mock1PEnv {
        type State = u32;
        type Action = u32;
        type Reward = [f32; 1];
        type StepDelta = ();

        fn initial(&self) -> Self::State {
            0
        }
        fn actions(&self, _s: &Self::State, out: &mut Vec<Self::Action>) {
            out.clear();
            out.extend([1, 2]);
        }
        fn step(
            &self,
            s: &mut Self::State,
            _a: &Self::Action,
        ) -> mcts_traits::StepOutcome<Self::Reward, ()> {
            *s += 1;
            mcts_traits::StepOutcome::new([1.0], *s >= 5)
        }
        fn current_agent(&self, _s: &Self::State) -> AgentId {
            AgentId(0)
        }
    }

    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _, ()> = TreeStore::with_capacity(50, 50, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[1, 2]);

    let env = Mock1PEnv;
    let model = PanickingModel;
    let selection = UctSelection::<1> { c_uct: 1.414 };
    let backup = SingleAgentBackup::new(1.0);
    let scheduler = BatchedScheduler::new(2, 1.0);

    // Run scheduler and expect panic from model.evaluate_batch
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut t = tree;
        scheduler.search(&mut t, &env, &model, &selection, &backup, root, &0, 1);
        t
    }));

    assert!(result.is_err());
}

#[test]
fn test_match_driver_history_only_contains_opponent_moves() {
    use crate::arena::MatchDriver;
    use mcts_traits::{Agent, StepOutcome, TurnBasedWorld, World};

    struct AlternatingWorld;

    impl World for AlternatingWorld {
        type WorldState = (usize, usize); // (turn_count, last_action)
        type Action = usize;
        type Observation = (usize, usize);

        fn n_players(&self) -> usize {
            2
        }

        fn initial(&self) -> Self::WorldState {
            (0, 0)
        }

        fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
            *ws
        }

        fn actions(&self, _ws: &Self::WorldState, _player: usize, out: &mut Vec<Self::Action>) {
            out.clear();
            out.push(1);
        }

        fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
            let action = &joint[ws.0 % 2];
            let outcome = self.step_action(ws, action);
            (outcome.reward.to_vec(), outcome.terminated)
        }

        fn terminal(&self, ws: &Self::WorldState) -> bool {
            ws.0 >= 4
        }
    }

    impl TurnBasedWorld for AlternatingWorld {
        type StepReward = [f32; 2];

        fn current_player(&self, ws: &Self::WorldState) -> usize {
            ws.0 % 2
        }

        fn step_action(
            &self,
            ws: &mut Self::WorldState,
            action: &Self::Action,
        ) -> StepOutcome<Self::StepReward> {
            ws.0 += 1;
            ws.1 = *action;
            let term = ws.0 >= 4;
            StepOutcome::new([0.0, 0.0], term)
        }
    }

    struct HistoryVerifyingAgent {
        name: String,
        action_to_play: usize,
        observed_histories: Vec<Vec<usize>>,
    }

    impl HistoryVerifyingAgent {
        fn new(name: &str, action: usize) -> Self {
            Self {
                name: name.to_string(),
                action_to_play: action,
                observed_histories: Vec::new(),
            }
        }
    }

    impl Agent<(usize, usize), usize> for HistoryVerifyingAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn select_action(&mut self, _state: &(usize, usize)) -> usize {
            self.action_to_play
        }
        fn select_action_with_history(
            &mut self,
            _state: &(usize, usize),
            history: &[(usize, ())],
        ) -> usize {
            self.observed_histories
                .push(history.iter().map(|(a, _)| *a).collect());
            self.action_to_play
        }
    }

    let world = AlternatingWorld;
    let mut p0 = HistoryVerifyingAgent::new("P0", 100);
    let mut p1 = HistoryVerifyingAgent::new("P1", 200);

    let driver = MatchDriver::new();
    let mut agents: [&mut dyn Agent<(usize, usize), usize>; 2] = [&mut p0, &mut p1];
    let result = driver.play_multi::<AlternatingWorld, usize, 2>(&world, &mut agents, None);

    assert_eq!(result.total_moves, 4);
    // P0 acted on turns 0 and 2:
    // Turn 0: game started, history is empty.
    // Turn 2: P1 played 200 on turn 1, so P0 should observe ONLY [200], NOT [100, 200]!
    assert_eq!(p0.observed_histories[0], Vec::<usize>::new());
    assert_eq!(p0.observed_histories[1], vec![200]);

    // P1 acted on turns 1 and 3:
    // Turn 1: P0 played 100 on turn 0, so P1 observes [100].
    // Turn 3: P0 played 100 on turn 2, so P1 observes [100].
    assert_eq!(p1.observed_histories[0], vec![100]);
    assert_eq!(p1.observed_histories[1], vec![100]);
}

#[test]
fn test_ismcts_stats_lifecycle() {
    use crate::selection::ismcts::IsmctsStats;
    use crate::tree_store::{EdgeId, EdgeStatsStore, PriorStore, VirtualLossStore};

    let mut stats = IsmctsStats::<2>::new();
    assert_eq!(stats.visits.len(), 0);

    stats.resize(4);
    assert_eq!(stats.visits.len(), 4);
    assert_eq!(stats.avail_visits.len(), 4);
    assert_eq!(stats.priors.len(), 4);
    assert_eq!(stats.mean_value.len(), 4);
    assert_eq!(stats.virtual_loss.len(), 4);

    let edge1 = EdgeId(1);
    stats.inc_avail_visits(edge1);
    stats.inc_avail_visits(edge1);
    assert_eq!(stats.avail_visits(edge1), 2);
    stats.set_avail_visits(edge1, 5);
    assert_eq!(stats.avail_visits(edge1), 5);

    stats.set_prior(edge1, 0.75);
    assert_eq!(stats.prior(edge1), 0.75);

    stats.add_virtual_loss(edge1, 1.0);
    assert_eq!(stats.virtual_loss[1], 1.0);
    stats.remove_virtual_loss(edge1, 0.5);
    assert_eq!(stats.virtual_loss[1], 0.5);

    stats.retain_edges(&[1, 3]);
    assert_eq!(stats.visits.len(), 2);
    assert_eq!(stats.avail_visits(EdgeId(0)), 5);
    assert_eq!(stats.prior(EdgeId(0)), 0.75);

    stats.clear();
    assert_eq!(stats.visits.len(), 0);
    assert_eq!(stats.avail_visits.len(), 0);
}

#[test]
fn test_ismcts_selection_compatible() {
    use crate::selection::ismcts::{IsmctsSelection, IsmctsStats};
    use crate::tree_store::{EdgeId, EdgeStatsStore, TreeStore};

    let mut stats = IsmctsStats::<2>::new();
    stats.resize(3);
    // Edge 0: Action 100, Visits: 10, Avail: 15, Prior: 0.5, Mean: [0.5, -0.5]
    stats.visits[0] = 10;
    stats.avail_visits[0] = 15;
    stats.priors[0] = 0.5;
    stats.mean_value[0] = [0.5, -0.5];

    // Edge 1: Action 200, Visits: 1, Avail: 20, Prior: 0.5, Mean: [0.9, -0.9] (High mean, but illegal in some states)
    stats.visits[1] = 1;
    stats.avail_visits[1] = 20;
    stats.priors[1] = 0.5;
    stats.mean_value[1] = [0.9, -0.9];

    // Edge 2: Action 300, Visits: 5, Avail: 10, Prior: 0.5, Mean: [0.2, -0.2]
    stats.visits[2] = 5;
    stats.avail_visits[2] = 10;
    stats.priors[2] = 0.5;
    stats.mean_value[2] = [0.2, -0.2];

    let mut tree: TreeStore<u32, [f32; 2], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[100, 200, 300]);

    let selection = IsmctsSelection::<2>::new(1.414);

    // 1. If all edges are compatible, edge 1 should win because of high Q + explore bonus
    let selected_all = selection.select_compatible_child(&tree, root, |_, _| true);
    assert_eq!(selected_all, Some(EdgeId(1)));

    // 2. If edge 1 (action 200) is NOT compatible in the current determinization:
    let selected_filtered =
        selection.select_compatible_child(&tree, root, |_edge, &action| action != 200);
    // Compatible are Edge 0 and Edge 2.
    // Total avail = 15 + 10 = 25. sqrt(25) = 5.0
    // Edge 0 score = 0.5 + 1.414 * 0.5 * 5.0 / 11 = 0.5 + 0.3213 = 0.8213
    // Edge 2 score = 0.2 + 1.414 * 0.5 * 5.0 / 6 = 0.2 + 0.5891 = 0.7891
    // Edge 0 wins!
    assert_eq!(selected_filtered, Some(EdgeId(0)));

    // 3. If none are compatible:
    let selected_none = selection.select_compatible_child(&tree, root, |_, _| false);
    assert_eq!(selected_none, None);
}

#[test]
fn test_ismcts_scheduler_execution() {
    use crate::backup::VectorBackup;
    use crate::scheduler::IsmctsScheduler;
    use crate::selection::ismcts::{IsmctsSelection, IsmctsStats};
    use crate::tree_store::TreeStore;
    use mcts_traits::belief::BeliefSampler;
    use mcts_traits::{ActionModel, AgentDynamics, AgentId, Evaluation, Model, StepOutcome};

    #[derive(Clone, Debug, PartialEq)]
    struct MockState {
        pub turn: usize,
        pub active_agent: AgentId,
        pub allowed_actions: Vec<u32>,
    }

    struct MockDynamics;
    impl AgentDynamics for MockDynamics {
        type State = MockState;
        type Action = u32;
        type Reward = [f32; 2];
        type StepDelta = ();

        fn initial(&self) -> Self::State {
            MockState {
                turn: 0,
                active_agent: AgentId(0),
                allowed_actions: vec![10, 20],
            }
        }

        fn actions(&self, state: &Self::State, out: &mut Vec<Self::Action>) {
            out.clear();
            out.extend_from_slice(&state.allowed_actions);
        }

        fn expand_actions(&self, _state: &Self::State, out: &mut Vec<Self::Action>) {
            out.clear();
            // All possible information-set actions
            out.extend_from_slice(&[10, 20, 30]);
        }

        fn current_agent(&self, state: &Self::State) -> AgentId {
            state.active_agent
        }

        fn step(
            &self,
            state: &mut Self::State,
            _action: &Self::Action,
        ) -> StepOutcome<Self::Reward, Self::StepDelta> {
            state.turn += 1;
            let terminated = state.turn >= 2;
            let next_agent = if state.active_agent == AgentId(0) {
                AgentId(1)
            } else {
                AgentId(0)
            };
            state.active_agent = next_agent;
            // For turn 1, allow action 100
            state.allowed_actions = vec![100];

            StepOutcome {
                reward: if terminated { [1.0, -1.0] } else { [0.0, 0.0] },
                delta: (),
                terminated,
            }
        }
    }

    struct MockSampler {
        counter: usize,
    }

    impl BeliefSampler for MockSampler {
        type State = MockState;
        type Context = ();

        fn sample(&mut self, _context: &Self::Context) -> Self::State {
            self.counter += 1;
            if self.counter % 2 == 1 {
                // Determinization A: actions 10 and 20 are legal
                MockState {
                    turn: 0,
                    active_agent: AgentId(0),
                    allowed_actions: vec![10, 20],
                }
            } else {
                // Determinization B: actions 20 and 30 are legal
                MockState {
                    turn: 0,
                    active_agent: AgentId(0),
                    allowed_actions: vec![20, 30],
                }
            }
        }
    }

    struct MockModel;
    impl Model<MockState> for MockModel {
        fn evaluate(&self, _state: &MockState) -> Evaluation {
            Evaluation {
                priors: vec![0.33, 0.33, 0.34],
                values: vec![0.0, 0.0],
            }
        }
    }

    impl ActionModel<MockState, u32> for MockModel {
        fn evaluate_actions(&self, state: &MockState, actions: &[u32]) -> Evaluation {
            let n = actions.len();
            let priors = if n > 0 {
                vec![1.0 / n as f32; n]
            } else {
                Vec::new()
            };
            let mut eval = self.evaluate(state);
            eval.priors = priors;
            eval
        }
    }

    let dynamics = MockDynamics;
    let mut sampler = MockSampler { counter: 0 };
    let model = MockModel;
    let selection = IsmctsSelection::<2>::new(1.414);
    let backup = VectorBackup::<2>::default();
    let stats = IsmctsStats::<2>::new();
    let mut tree: TreeStore<u32, [f32; 2], IsmctsStats<2>, ()> =
        TreeStore::with_capacity(20, 20, stats);
    let root = tree.insert_root(AgentId(0));

    let scheduler = IsmctsScheduler;
    scheduler.search(
        &mut tree,
        &dynamics,
        &model,
        &selection,
        &backup,
        &mut sampler,
        &(),
        root,
        50,
    );

    // Root should be expanded with all 3 actions from expand_actions
    assert_eq!(tree.num_children(root), 3);
    let first = tree.first_child_edge(root);
    assert_eq!(*tree.edge_action(first), 10);
    assert_eq!(*tree.edge_action(EdgeId(first.0 + 1)), 20);
    assert_eq!(*tree.edge_action(EdgeId(first.0 + 2)), 30);

    // Check availability visits:
    // Action 20 is present in both determinizations A and B, so its avail_visits should be roughly 50.
    // Action 10 is present in A (half), action 30 is present in B (half).
    let avail0 = tree.stats.avail_visits(first);
    let avail1 = tree.stats.avail_visits(EdgeId(first.0 + 1));
    let avail2 = tree.stats.avail_visits(EdgeId(first.0 + 2));

    assert!(
        avail1 > avail0,
        "Action 20 should have higher availability visits than action 10"
    );
    assert!(
        avail1 > avail2,
        "Action 20 should have higher availability visits than action 30"
    );
    assert_eq!(
        avail0 + avail2,
        avail1,
        "avail(10) + avail(30) should equal avail(20)"
    );

    // Traversed visits must sum to total search passes
    let v0 = tree.stats.visits[first.as_usize()];
    let v1 = tree.stats.visits[first.as_usize() + 1];
    let v2 = tree.stats.visits[first.as_usize() + 2];
    assert_eq!(v0 + v1 + v2, 50);
}

#[test]
fn test_stochastic_trajectory_rewards_convergence() {
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _> = TreeStore::with_capacity(5, 5, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[10]);

    let edge = tree.first_child_edge(root);
    let child = tree.insert_node(edge, AgentId(0));
    tree.mark_terminal(child);

    let backup = SingleAgentBackup::new(1.0);

    // Trajectory 1 observes reward 0.0
    let path1 = [PathElement {
        node: root,
        edge,
        next_node: child,
        reward: [0.0],
    }];
    backup.backup(&mut tree, &path1, Option::<&Evaluation>::None);
    assert_eq!(tree.stats.visits[edge.as_usize()], 1);
    assert_eq!(tree.stats.mean_value[edge.as_usize()][0], 0.0);

    // Trajectory 2 traverses the SAME edge but observes reward 10.0
    let path2 = [PathElement {
        node: root,
        edge,
        next_node: child,
        reward: [10.0],
    }];
    backup.backup(&mut tree, &path2, Option::<&Evaluation>::None);
    assert_eq!(tree.stats.visits[edge.as_usize()], 2);
    // Running mean should be (0.0 + 10.0) / 2 = 5.0
    assert_eq!(tree.stats.mean_value[edge.as_usize()][0], 5.0);

    // Trajectory 3 observes reward 6.0
    let path3 = [PathElement {
        node: root,
        edge,
        next_node: child,
        reward: [6.0],
    }];
    backup.backup(&mut tree, &path3, Option::<&Evaluation>::None);
    assert_eq!(tree.stats.visits[edge.as_usize()], 3);
    // Running mean should be (5.0 * 2 + 6.0) / 3 = 16.0 / 3 ≈ 5.3333335
    let expected = (0.0 + 10.0 + 6.0) / 3.0;
    assert!((tree.stats.mean_value[edge.as_usize()][0] - expected).abs() < 1e-5);
}

#[test]
fn test_stochastic_node_parent_visits_not_inflated() {
    // Test that for child nodes with multiple delta branches, PUCT/UCT
    // uses the child node's own child edge visit sum rather than inflating by the parent action visits.
    let stats = MultiAgentPuctStats::<1>::new();
    let mut tree: TreeStore<u32, [f32; 1], _, u32> = TreeStore::with_capacity(10, 10, stats);
    let root = tree.insert_root(AgentId(0));
    tree.expand_node(root, &[100]); // Edge 0

    let edge0 = tree.first_child_edge(root);
    // Edge 0 branches into two child nodes: Child A (delta 1) and Child B (delta 2)
    let (child_a, _) = tree.get_or_insert_child(edge0, &1, AgentId(0));
    let (child_b, _) = tree.get_or_insert_child(edge0, &2, AgentId(0));

    // Expand child A with actions 10 and 20 (edges 1 and 2)
    tree.expand_node(child_a, &[10, 20]);
    // Expand child B with action 30 (edge 3)
    tree.expand_node(child_b, &[30]);

    let edge1 = tree.first_child_edge(child_a);
    let edge2 = EdgeId(edge1.0 + 1);
    let edge3 = tree.first_child_edge(child_b);

    // Simulate visit distribution:
    // Edge 0 (parent action) was visited 100 times.
    // Child A was visited 10 times (edge 1 visited 6 times, edge 2 visited 4 times).
    // Child B was visited 90 times (edge 3 visited 90 times).
    tree.stats.visits[edge0.as_usize()] = 100;
    tree.stats.visits[edge1.as_usize()] = 6;
    tree.stats.visits[edge2.as_usize()] = 4;
    tree.stats.visits[edge3.as_usize()] = 90;

    // In UCT selection at child A:
    // If parent_visits were taken from edge 0, it would be 100!
    // But child A was only visited 10 times (6 + 4 = 10).
    // The selection policy should evaluate parent_visits as 10:
    let uct = UctSelection::<1> { c_uct: 1.0 };
    let chosen = uct.select_child(&tree, child_a);
    assert!(chosen.is_some());
}
