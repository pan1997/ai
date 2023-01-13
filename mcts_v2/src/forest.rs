use std::collections::BTreeMap;

use lib_v2::utils::RunningAverage;

// an arena based tree
// does not support deletion of nodes

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeId(usize);

#[derive(Debug)]
pub struct Node<A, O> {
  pub(crate) actions: BTreeMap<A, ActionInfo>,
  // index to children
  children: BTreeMap<O, NodeId>,
  value: RunningAverage,
  select_count: u32,
}

#[derive(Debug)]
pub(crate) struct ActionInfo {
  action_reward: RunningAverage,
  value_of_next_state: RunningAverage,
  select_count: u32,
  static_policy_score: f32,
}

#[derive(Debug)]
pub(crate) struct Forest<A, O> {
  // one rooted tree for each Agent
  nodes: Vec<Node<A, O>>,
  roots: Vec<NodeId>,
}

impl<A, O> Forest<A, O>
where
  O: Ord + Clone,
{
  pub fn new(capacity: usize) -> Self {
    Self {
      nodes: Vec::with_capacity(capacity),
      roots: Vec::new(),
    }
  }

  pub fn new_root(&mut self) -> NodeId {
    let r = self.new_node();
    self.roots.push(r);
    r
  }

  pub fn roots(&self) -> Vec<NodeId> {
    self.roots.clone()
  }

  fn new_node(&mut self) -> NodeId {
    let id = self.nodes.len();
    self.nodes.push(Node::new());
    NodeId(id)
  }

  pub(crate) fn node(&self, node_id: NodeId) -> &Node<A, O> {
    &self.nodes[node_id.0]
  }

  pub(crate) fn node_mut(&mut self, node_id: NodeId) -> &mut Node<A, O> {
    &mut self.nodes[node_id.0]
  }

  pub(crate) fn get_id_of_child(&mut self, node_id: NodeId, o: &O) -> NodeId {
    if !self.nodes[node_id.0].children.contains_key(o) {
      let new_node_id = self.new_node();
      self.nodes[node_id.0]
        .children
        .insert(o.clone(), new_node_id);
      new_node_id
    } else {
      *self.nodes[node_id.0].children.get(o).unwrap()
    }
  }
}

impl<A, O> Node<A, O> {
  fn new() -> Self {
    Self {
      actions: BTreeMap::new(),
      children: BTreeMap::new(),
      select_count: 0,
      value: RunningAverage::new(),
    }
  }
  pub(crate) fn select_count(&self) -> u32 {
    self.select_count
  }

  pub(crate) fn increment_select_count(&mut self) {
    self.select_count += 1;
  }
}

impl<A: Ord, O> Node<A, O> {
  pub(crate) fn create_actions(&mut self, actions: Vec<A>) {
    debug_assert!(self.actions.is_empty(), "recreating actions");
    actions.into_iter().for_each(|action| {
      self.actions.insert(
        action,
        ActionInfo {
          action_reward: RunningAverage::new(),
          value_of_next_state: RunningAverage::new(),
          select_count: 0,
          static_policy_score: 1.0,
        },
      );
    });
  }
}

impl ActionInfo {
  pub(crate) fn select_count(&self) -> u32 {
    self.select_count
  }

  pub(crate) fn increment_select_count(&mut self) {
    self.select_count += 1;
  }

  pub(crate) fn value(&self) -> f32 {
    self.action_reward.value()
  }
}
