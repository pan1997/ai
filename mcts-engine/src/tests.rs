use crate::backup::{BackupPolicy, SingleAgentBackup, VectorBackup};
use crate::scheduler::{BatchedScheduler, MultiGameScheduler, SequentialScheduler};
use crate::selection::{MultiAgentPuctSelection, MultiAgentPuctStats, SelectionPolicy, UctSelection};
use crate::tree_store::{EdgeId, TreeStore};
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

