use std::{
  collections::BTreeMap,
  fmt::{Debug, Display},
};

use lib::utils::RunningAverage;
pub mod render;

// an arena based tree
// does not support deletion of nodes

#[derive(Clone, Copy, Debug)]
pub struct NodeId(usize);

#[derive(Debug)]
pub struct Node<A, O> {
  // todo: remove
  id: usize,
  actions_created: bool,
  pub(crate) actions: BTreeMap<A, ActionInfo>,
  // index to children
  children: BTreeMap<O, NodeId>,
  pub(crate) value: RunningAverage,
  select_count: u32,
}

#[derive(Debug)]
pub(crate) struct ActionInfo {
  pub(crate) action_reward: RunningAverage,
  pub(crate) value_of_next_state: RunningAverage,
  select_count: u32,
  pub(crate) static_policy_score: f32,
}

#[derive(Debug)]
pub struct Forest<A, O> {
  // one rooted tree for each Agent
  nodes: Vec<Node<A, O>>,

  // the order doesn't change
  roots: Vec<NodeId>,
}

impl<A, O> Forest<A, O>
where
  // todo remove debug
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
    self.nodes[id].id = id;
    NodeId(id)
  }

  pub(crate) fn get_id_of_child(&mut self, node_id: NodeId, o: &O) -> NodeId {
    //print!("fetching child {} of {}:", o, node_id.0);
    if !self.nodes[node_id.0].children.contains_key(o) {
      let new_node_id = self.new_node();
      self.nodes[node_id.0]
        .children
        .insert(o.clone(), new_node_id);
      //println!("new {}", new_node_id.0);
      new_node_id
    } else {
      let r = *self.nodes[node_id.0].children.get(o).unwrap();
      //println!("old {}", r.0);
      r
    }
  }
}

impl<A, O> Node<A, O> {
  fn new() -> Self {
    Self {
      id: 0,
      actions_created: false,
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

  pub(crate) fn actions_created(&self) -> bool {
    self.actions_created
  }
}

impl<A: Ord, O> Forest<A, O> {
  pub(crate) fn node(&self, node_id: NodeId) -> &Node<A, O> {
    &self.nodes[node_id.0]
  }

  pub(crate) fn node_mut(&mut self, node_id: NodeId) -> &mut Node<A, O> {
    &mut self.nodes[node_id.0]
  }
}

impl<A: Ord, O> Node<A, O> {
  pub(crate) fn create_actions(&mut self, actions: Vec<A>) {
    //println!("Creating actions on node:{}", self.id);
    debug_assert!(!self.actions_created, "recreating actions");
    self.actions_created = true;
    let s = 1.0 / actions.len() as f32;
    actions.into_iter().for_each(|action| {
      self.actions.insert(
        action,
        ActionInfo {
          action_reward: RunningAverage::new(),
          value_of_next_state: RunningAverage::new(),
          select_count: 0,
          static_policy_score: s,
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
    self.action_reward.value() + self.value_of_next_state.value()
  }
}

impl<A: Display, O: Display> Display for Node<A, O> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Node {{\"id\": {}", self.id)?;
    let actions: Vec<_> = self.actions.keys().map(|k| k.to_string()).collect();
    write!(f, "\"actions\": {:?}", actions)?;
    let children: Vec<_> = self
      .children
      .iter()
      .map(|(k, v)| (k.to_string(), v))
      .collect();
    write!(
      f,
      ", \"children:\": {:?}, \"select_count\": {}",
      children, self.select_count
    )?;
    write!(f, "}}")?;
    Ok(())
  }
}
