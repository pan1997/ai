use crate::util::Average;
use std::collections::BTreeMap;

pub(crate) struct Node<A, O> {
  pub(crate) actions: BTreeMap<A, ActionInfo>,
  children: BTreeMap<O, Node<A, O>>,

  // value of this state
  value: Average,
  select_count: u32,
}

pub(crate) struct ActionInfo {
  action_reward: Average,
  value_of_next_state: Average,

  select_count: u32,
}

impl<A, O> Node<A, O> {
  fn new() -> Self {
    Self {
      actions: BTreeMap::new(),
      children: BTreeMap::new(),
      value: Average::new(),
      select_count: 0,
    }
  }
}
