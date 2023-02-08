

/*
 open queue contains all (node, state) tuples ready to select
 selected queue contains all (node, state, action) tuples
 applied queue contains all (node, state) tuples
 backprop queue contains (node, state) tuples ready to init/backprop

 threadpool 1 -> 
   pop one (node, state) from open queue and apply tree policy to obtain action
   increment select count of node
   increment select count of action in node
   push (node, state, action) into selected queue
  
 threadpool 2 ->
   pop a batch of (node, state, action)'s from open.
   generate next state's for each element in batch, by applying the corresponding action
   push all (node, new_state) to applied
  
 threadpool 3 ->
   pop one (node, state) from applied
   get the

  
  open: Queue<(node, state)>
  selected: Queue<(node, state, action)>

  worker 1 ->
    to_init: (Vec<node>, Vec<state>)
    
    loop:
      pop one (node, state) from open (blocking if open is empty) and lock the node
      if it doesn't have action info:
        create action_info on this node using the legal actions on state 
        unlock node

        // a node is pushed to init only in the iteration in which it's action info
        // was created. in between the action info being created and the node being
        // init, it can still be selected by other threads (or the same thread)
        
        push it in to_init
        if to_init size limit is reached:
          apply batch_estimator to all states in to_init, and obtain pi and v
          update the nodes's static policy estimates with pi
          backprop v for all the nodes
          clear to_init
      else:
        select action for (node, state) using tree policy
        increment select count for node
        increment select count for action in node
        unlock node
        push (node, state, action) to selected
        if selected size limit is reached:
          apply batch apply_action to all (state, action) pairs in selected
          obtain the observations o and rewards for all transitions
          for each (node, obervation):
            new_node = get_or_create_child of node for observation
            push (new_node, state) to open
 */

use std::{collections::BTreeMap, sync::{Mutex, RwLock}};

use lib::utils::RunningAverage;
use std::sync::Arc;


struct Node<A, O> {
  // children are behind a mutex
  children: RwLock<BTreeMap<O, Arc<Node<A, O>>>>,
  // data is behind rw lock
  // is none when the node has never been expanded
  data: RwLock<Option<NodeData<A>>>
}

struct NodeData<A> {
  action_data: BTreeMap<A, ActionInfo>,
  select_count: u32,
  value: RunningAverage
}

struct ActionInfo {
  action_reward: RunningAverage,
  value_of_next_state: RunningAverage,
  select_count: u32,
  static_policy_score: f32,
}

impl<A,O> Node<A, O> {
  fn new() -> Self {
    Self { 
      children: RwLock::new(BTreeMap::new()), 
      data: RwLock::new(None) 
    }
  }
}

impl<A, O: Ord + Clone> Node<A, O>{
  fn get_child(&self, o: O) -> Arc<Self> {
    let guard = self.children.read().unwrap();
    if !guard.contains_key(&o) {
      drop(guard);
      let mut lock = self.children.write().unwrap();
      // we need to check once again if the child hasn't been created by a 
      // competing thread
      if !lock.contains_key(&o) {
        let result = Arc::new(Node::new());
        lock.insert(o.clone(), result.clone());
        result
      } else {
        lock[&o].clone()
      }
    } else {
      guard[&o].clone()
    }
  }
}

#[cfg(test)]
mod test {
  use super::Node;

  #[test]
  fn t1() {
  }
}