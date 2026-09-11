use crate::backup::{BackupPolicy, PathElement, SingleAgentBackup, VectorBackup};
use crate::scheduler::{BatchedScheduler, MultiGameScheduler, SequentialScheduler};
use crate::selection::{
    GumbelPuctSelection, MultiAgentPuctSelection, MultiAgentPuctStats, SelectionPolicy,
    UctSelection,
};
use crate::tree_store::{EdgeId, NodeId, NodeStatus, PriorStore, TreeStore, VirtualLossStore};
use mcts_traits::{AgentId, BatchedModel, Evaluation, GraphEnv, Model};
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
    let _child = tree.insert_node(edge, AgentId(0));

    let backup = SingleAgentBackup::new(0.9); // gamma = 0.9
    let path = [crate::backup::PathElement { node: root, edge }];
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
        },
        PathElement {
            node: node1,
            edge: edge1,
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
    assert!((sum - 1.0).abs() < 1e-4, "Dirichlet noise should preserve normalization: sum={sum}");

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
    assert!((tree_sum - 1.0).abs() < 1e-4, "Tree priors should sum to 1.0: sum={tree_sum}");
}


