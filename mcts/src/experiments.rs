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


 //open: Queue<(node, state)>
 current_trajectory: Trajectory
 current_state
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
           lock node
           new_node = get_or_create_child of node for observation
           unlock node
           push (new_node, state) to open
*/

use std::{
  collections::{BTreeMap, VecDeque},
  marker::PhantomData,
  sync::{Arc, Mutex, RwLock},
};

use lib::{
  utils::{Bounds, RunningAverage},
  MctsProblem,
};

use crate::{bandits::Bandit, SearchLimit, Expansion};

/*
struct Node<A, O> {
  // children are behind a mutex
  children: RwLock<BTreeMap<O, Arc<Node<A, O>>>>,
  // data is behind rw lock
  // is none when the node has never been expanded
  data: RwLock<Option<NodeData<A, O>>>
}*/

type NodePtr<A, O> = Arc<Mutex<NodeData<A, O>>>;
type CompositeNodePtr<A, O> = Vec<NodePtr<A, O>>;

struct NodeData<A, O> {
  children: BTreeMap<O, NodePtr<A, O>>,
  select_count: u32,
  value: RunningAverage,
  actions_created: bool,
  action_data: BTreeMap<A, ActionInfo>,
}

struct ActionInfo {
  action_reward: RunningAverage,
  value_of_next_state: RunningAverage,
  select_count: u32,
  static_policy_score: f32,
}

impl<A, O> NodeData<A, O> {
  fn new() -> Self {
    Self {
      children: BTreeMap::new(),
      select_count: 0,
      value: RunningAverage::new(),
      actions_created: false,
      action_data: BTreeMap::new(),
    }
  }

  fn has_actions_created(&self) -> bool {
    self.actions_created
  }

  fn create_actions(&mut self, actions: Vec<A>)
  where
    A: Ord,
  {
    self.actions_created = true;
    self.action_data = BTreeMap::from_iter(actions.into_iter().map(|a| {
      (
        a,
        ActionInfo {
          action_reward: RunningAverage::new(),
          value_of_next_state: RunningAverage::new(),
          select_count: 0,
          static_policy_score: 0.0,
        },
      )
    }));
  }
}

impl<A, O: Ord + Clone> NodeData<A, O> {
  fn get_child(&mut self, o: O) -> NodePtr<A, O> {
    //let entry =
    self
      .children
      .entry(o)
      .or_insert_with(|| Arc::new(Mutex::new(NodeData::new())))
      .clone()
    /*

    //let guard = self.children.read().unwrap();
    if !self.children.contains_key(&o) {
      // we need to check once again if the child hasn't been created by a
      // competing thread
      let result = Arc::new(Mutex::new(NodeData::new()));
      self.children.insert(o.clone(), result.clone());
      result
    } else {
      self.children[&o].clone()
    }*/
  }
}

pub struct Search<P: MctsProblem, B, E> {
  problem: Arc<P>,
  b_state: Arc<P::BeliefState>,
  block_size: usize,
  limit: SearchLimit,
  bandit_policy: B,
  static_estimator: E,
  root: CompositeNodePtr<P::Action, P::Observation>,
}

type AgentIndex = usize;

#[derive(Clone)]
struct Trajectory<A, O>
where
  A: Clone,
  O: Clone,
{
  current: CompositeNodePtr<A, O>,
  history: Vec<(CompositeNodePtr<A, O>, Vec<f32>, AgentIndex, A)>,
}

impl<P, B, E> Search<P, B, E>
where
  P: MctsProblem,
{
  fn _backpropogate(
    &self,
    _trajectory: &Trajectory<P::Action, P::Observation>,
    mut _values: Vec<f32>,
  ) {
  }
}

struct Worker<A: Clone, O: Clone, S> {
  open_states: Vec<S>,
  open_trajectories: Vec<Trajectory<A, O>>,

  closed_states: Vec<S>,
  closed_trajectories: Vec<Trajectory<A, O>>,
  // each worker has independent bounds
  score_bounds: Vec<Bounds>,
}

impl<A: Clone, O: Clone, S> Worker<A, O, S> {
  fn _start<P, B, E>(&mut self, search: &Search<P, B, E>)
  where
    P: MctsProblem<Action = A, Observation = O, HiddenState = S>,
    B: Bandit<S, A, O>,
    A: Default, // todo: remove
    A: Ord,
    O: Ord + Clone,
    S: Clone, // todo: remove
    E: Expansion<P>
  {
    loop {
      let mut open_actions = vec![Default::default(); self.open_states.len()];
      let mut open_current_agent_ix = vec![0; self.open_states.len()];
      for index in 0..self.open_trajectories.len() {
        if search.problem.check_terminal(&self.open_states[index]) {
          search._backpropogate(&self.open_trajectories[index], vec![]);
          self.open_states[index] = search.problem.sample_h_state(&search.b_state);
          self.open_trajectories[index].reset(search.root.clone());

          // explain
          // we cannot continue, because of batching
          //continue 'inner;
        }
        // if root is non terminal, it's guaranteed that open[index] is now non terminal

        let mut node_data_guard = self.open_trajectories[index].current
          [search.problem.agent_to_act(&self.open_states[index]).into() as usize]
          .lock()
          .unwrap();
        let mut node_data_guard = if !node_data_guard.has_actions_created() {
          node_data_guard.create_actions(search.problem.legal_actions(&self.open_states[index]));
          drop(node_data_guard);

          self.closed_states.push(self.open_states[index].clone());
          self
            .closed_trajectories
            .push(self.open_trajectories[index].clone());

          if self.closed_states.len() >= search.block_size {
            let (values, static_policies) = search.static_estimator.block_expand(&search.problem, &self.closed_states);
            for ix in 0..self.closed_states.len() {
              // todo: remove this clone
              search._backpropogate(&self.closed_trajectories[ix], values[ix].clone());
              let current_agent_ix = search.problem.agent_to_act(&self.closed_states[ix]).into() as usize;
              let mut guard = self.closed_trajectories[ix].current[current_agent_ix].lock().unwrap();
              for (a, p) in static_policies[ix].iter() {
                guard.action_data.get_mut(a).unwrap().static_policy_score = *p;
              }
            }
          }

          self.open_states[index] = search.problem.sample_h_state(&search.b_state);
          self.open_trajectories[index].reset(search.root.clone());
          // explain
          self.open_trajectories[index].current
            [search.problem.agent_to_act(&self.open_states[index]).into() as usize]
            .lock()
            .unwrap()
        } else {
          node_data_guard
        };

        // actions have been created for this node,
        // it might not have a value and static policy
        // assigned yet

        let current_agent_ix = search.problem.agent_to_act(&self.open_states[index]).into() as usize;
        
        //search.bandit_policy.select(&self.open_states[index], &self.open_trajectories[index].current[current_agent_ix], &self.score_bounds[current_agent_ix]);
        let selected_action: A = Default::default();
        node_data_guard.select_count += 1;
        node_data_guard.action_data.get_mut(&selected_action).unwrap().select_count += 1;
        open_actions[index] = selected_action;
        open_current_agent_ix[index] = current_agent_ix;
      }
      // apply actions

      let action_result = search.problem.apply_action_batched(&mut self.open_states, &open_actions);
      // descend tree
      for index in 0..self.open_states.len() {
        let old_node = self.open_trajectories[index].current.clone();
        // update current 
        for (ix, n) in self.open_trajectories[index].current.iter_mut().enumerate() {
           let mut guard = n.lock().unwrap();
           let child_ix = guard.get_child(action_result[index][ix].1.clone());
           drop(guard);
           *n = child_ix;
        }
        let rewards = action_result[index].iter().map(|(a, _)| *a).collect();
        self.open_trajectories[index].history.push((old_node, rewards, open_current_agent_ix[index], open_actions[index].clone()));

      }
    }
  }
}

impl<A: Clone, O: Clone> Trajectory<A, O> {
  fn reset(&mut self, node: CompositeNodePtr<A, O>) {
    self.history.clear();
    self.current = node;
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn t1() {}
}
