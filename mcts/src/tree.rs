use crate::util::Average;
use petgraph::{graph::NodeIndex, Graph};
use std::cell::{Cell, UnsafeCell};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct Node<A, O> {
  pub(crate) actions: BTreeMap<A, ActionInfo>,
  // make thread safe
  children: UnsafeCell<BTreeMap<O, Node<A, O>>>,

  // value of this state
  pub(crate) value: Average,
  select_count: Cell<u32>,
}

#[derive(Debug)]
pub(crate) struct ActionInfo {
  pub(crate) action_reward: Average,
  pub(crate) value_of_next_state: Average,

  select_count: Cell<u32>,
}

impl<A: Ord + Clone, O: Ord> Node<A, O> {
  pub fn new(a: &[A]) -> Self {
    Self {
      actions: BTreeMap::from_iter(a.iter().map(|a| (a.clone(), ActionInfo::new()))),
      children: UnsafeCell::new(BTreeMap::new()),
      value: Average::new(),
      select_count: Cell::new(0),
    }
  }

  pub(crate) fn increment_select_count(&self) {
    self.select_count.set(self.select_count() + 1);
  }

  pub(crate) fn select_count(&self) -> u32 {
    self.select_count.get()
  }

  pub(crate) fn next_node(&self, o: &O) -> Option<&Node<A, O>> {
    unsafe { (*self.children.get()).get(o) }
  }

  pub(crate) fn create_new_node(&self, o: O, actions: Vec<A>) {
    let old = unsafe { (&mut *self.children.get()).insert(o, Node::new(&actions)) };
    debug_assert!(old.is_none(), "reinserting")
  }
}

impl ActionInfo {
  fn new() -> Self {
    Self {
      action_reward: Average::new(),
      value_of_next_state: Average::new(),
      select_count: Cell::new(0),
    }
  }

  pub(crate) fn increment_select_count(&self) {
    self.select_count.set(self.select_count() + 1);
  }

  pub(crate) fn select_count(&self) -> u32 {
    self.select_count.get()
  }
}

pub fn render_petg<A: Ord + Clone, O: Clone + Ord>(
  node: &Node<A, O>,
  graph: &mut Graph<(f32, u32), O>,
  theta: u32,
  depth: u32,
) -> petgraph::graph::NodeIndex {
  let n = graph.add_node((node.value.mean(), node.select_count()));
  if depth > 0 && node.select_count() > theta {
    let children = unsafe { &*node.children.get() };
    for o in children.keys() {
      let t_ix = render_petg(&children[o], graph, theta, depth - 1);
      graph.add_edge(n, t_ix, o.clone());
    }
  }
  n
}
