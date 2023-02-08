use std::{
  fmt::Debug,
  sync::{Arc, RwLock},
};

use lib::{utils::Bounds, MctsProblem};

use crate::{
  bandits::Bandit,
  forest::{Forest, NodeId},
  Expansion, SearchLimit,
};

pub struct Search<P: MctsProblem, B, E> {
  problem: Arc<P>,
  b_state: Arc<P::BeliefState>,
  // todo: remove pub
  pub forest: RwLock<Forest<P::Action, P::Observation>>,
  block_size: u32,
  limit: SearchLimit,
  bandit_policy: B,
  static_estimator: E,
  score_bounds: RwLock<Vec<Bounds>>,
}

#[derive(Clone)]
pub struct Worker<S, A: Clone> {
  states_in_flight: Vec<S>,
  trajectories_in_flight: Vec<Trajectory<A>>,

  states_awaiting_expansion: Vec<S>,
  trajectories_awaiting_expansion: Vec<Trajectory<A>>,

  // expansion backprops by itself.
  // trajectories that are terminated during select phase are
  // queued here. these have terminal value zero, as they end
  // in terminal nodes
  trajectories_awaiting_backprop: Vec<Trajectory<A>>,
}

#[derive(Clone)]
pub struct Trajectory<A: Clone> {
  // one node in each player's tree
  pub(crate) current_: Vec<NodeId>,

  // nodeId, emitted reward and the selected action (along with the index of agent)
  branch: Vec<(Vec<(NodeId, f32)>, (usize, A))>,
}

impl<P: MctsProblem, B, E> Search<P, B, E>
where
  B: Bandit<P::HiddenState, P::Action, P::Observation>,
  // todo: remove this requirement
  P::HiddenState: Clone,
  E: Expansion<P>,
{
  pub fn new(
    problem: Arc<P>,
    b_state: Arc<P::BeliefState>,
    block_size: u32,
    limit: SearchLimit,
    bandit_policy: B,
    node_init: E,
  ) -> Self {
    let mut forest = Forest::new(800);
    let agent_count = problem.agents().len();
    for _ in 0..agent_count {
      forest.new_root();
    }

    Search {
      problem,
      b_state,
      forest: RwLock::new(forest),
      block_size,
      limit,
      bandit_policy,
      score_bounds: RwLock::new(vec![Bounds::new(); agent_count]),
      static_estimator: node_init,
    }
  }

  pub fn start(&self, worker: &mut Worker<P::HiddenState, P::Action>) {
    // initialize root node if needed
    {
      let mut guard = self.forest.write().unwrap();
      for (state, trajectory) in worker
        .states_in_flight
        .iter()
        .zip(worker.trajectories_in_flight.iter())
      {
        let current_agent_ix = self.problem.agent_to_act(state).into() as usize;
        let node_id = trajectory.current_[current_agent_ix];
        let node = guard.node_mut(node_id);
        if !node.actions_created() {
          node.create_actions(self.problem.legal_actions(state));
          let (_, p) = self.static_estimator.expand(&self.problem, state);
          for (a, pa) in p {
            node.actions.get_mut(&a).unwrap().static_policy_score = pa;
          }
        }
      }
    }

    loop {
      //println!("worker trajectories: {:?}", worker.trajectories_in_flight);
      // select actions
      let agents_and_actions: Vec<_> = {
        //let guard = self.forest.read().unwrap();
        let bounds_guard = self.score_bounds.read().unwrap();
        // check if search budget remains
        let select_count_root = {
          let guard = self.forest.read().unwrap();
          guard.node(guard.roots()[0]).select_count()
        };
        if !self.limit.more(select_count_root) {
          return;
        }
        worker
          .trajectories_in_flight
          .iter_mut()
          .zip(worker.states_in_flight.iter_mut())
          .map(|(trajectory, state)| {
            let guard = self.forest.read().unwrap();
            if self.problem.check_terminal(&state) {
              worker
                .trajectories_awaiting_backprop
                .push(trajectory.clone());
              *state = self.problem.sample_h_state(&self.b_state);
              self.restart_trajectory(&guard, trajectory);
            }
            // its guaranteed that the state is not terminal
            if !guard
              .node(trajectory.current_[self.problem.agent_to_act(state).into() as usize])
              .actions_created()
            {
              worker.states_awaiting_expansion.push(state.clone());
              worker
                .trajectories_awaiting_expansion
                .push(trajectory.clone());

              *state = self.problem.sample_h_state(&self.b_state);
              self.restart_trajectory(&guard, trajectory);
            }

            // need this before applying action to states that can change the aganet to act
            let current_agent_ix = self.problem.agent_to_act(state).into() as usize;
            // its guaranteed that the trajectory is not terminal
            (
              current_agent_ix,
              self.bandit_policy.select(
                state,
                guard.node(trajectory.current_[current_agent_ix]),
                &bounds_guard[current_agent_ix],
              ),
            )
          })
          .collect()
      };
      let (agents, actions): (Vec<_>, Vec<_>) = agents_and_actions.into_iter().unzip();

      // apply_actions
      let outcomes = self
        .problem
        .apply_action_batched(&mut worker.states_in_flight, &actions);

      let expansion_result = if worker.states_awaiting_expansion.len() >= self.block_size as usize {
        Some(
          self
            .static_estimator
            .block_expand(&self.problem, &worker.states_awaiting_expansion),
        )
      } else {
        None
      };

      // process backprop for trajectories that were terminated during select
      // expand nodes that are to be expanded
      // descend tree
      {
        let mut guard = self.forest.write().unwrap();
        let mut bound_guard = self.score_bounds.write().unwrap();
        for trajectory in worker.trajectories_awaiting_backprop.iter() {
          self.backpropogate(
            &mut guard,
            &mut bound_guard,
            trajectory,
            vec![0.0; trajectory.current_.len()],
          );
        }
        worker.trajectories_awaiting_backprop.clear();

        // process expansion queue
        for (state, trajectory) in worker
          .states_awaiting_expansion
          .iter_mut()
          .zip(worker.trajectories_awaiting_expansion.iter_mut())
        {
          let current_agent_ix = self.problem.agent_to_act(state).into() as usize;
          if !guard
            .node(trajectory.current_[current_agent_ix])
            .actions_created()
          {
            let node = guard.node_mut(trajectory.current_[current_agent_ix]);
            node.create_actions(self.problem.legal_actions(state));
          }
        }

        if expansion_result.is_some() {
          let (v, p) = expansion_result.unwrap();
          for ((state, trajectory), (value, static_policy)) in worker
            .states_awaiting_expansion
            .iter()
            .zip(worker.trajectories_awaiting_expansion.iter())
            .zip(v.into_iter().zip(p.into_iter()))
          {
            self.backpropogate(&mut guard, &mut bound_guard, trajectory, value);
            let current_agent_ix = self.problem.agent_to_act(state).into() as usize;
            for (a, pa) in static_policy {
              guard
                .node_mut(trajectory.current_[current_agent_ix])
                .actions
                .get_mut(&a)
                .unwrap()
                .static_policy_score = pa;
            }
          }
          worker.trajectories_awaiting_expansion.clear();
          worker.states_awaiting_expansion.clear();
        }

        worker
          .trajectories_in_flight
          .iter_mut()
          .zip(worker.states_in_flight.iter())
          .zip(
            actions
              .into_iter()
              .zip(agents.into_iter())
              .zip(outcomes.into_iter()),
          )
          .for_each(
            |((trajectory, state), ((action, agent_ix), outcomes_and_rewards))| {
              // increment select_counts
              // descend nodes

              // the state here has the action applied to it, but the trajectory's current points to the old one

              let mut children_ix = Vec::with_capacity(trajectory.current_.len());
              let mut branch_entry = Vec::with_capacity(trajectory.current_.len());
              for (ix, node_id) in trajectory.current_.iter().enumerate() {
                {
                  let node = guard.node_mut(*node_id);
                  node.increment_select_count();
                  if ix == agent_ix {
                    node
                      .actions
                      .get_mut(&action)
                      .unwrap()
                      .increment_select_count();
                  }
                }
                children_ix.push(guard.get_id_of_child(*node_id, &outcomes_and_rewards[ix].1));
                branch_entry.push((*node_id, outcomes_and_rewards[ix].0));
              }
              trajectory.current_ = children_ix;
              trajectory.branch.push((branch_entry, (agent_ix, action)));
            },
          );
      }
    }
  }

  fn backpropogate(
    &self,
    forest: &mut Forest<P::Action, P::Observation>,
    bounds: &mut Vec<Bounds>,
    trajectory: &Trajectory<P::Action>,
    mut values: Vec<f32>,
  ) {
    // add this value sample to the trajectory's current nodes
    for (ix, nid) in trajectory.current_.iter().enumerate() {
      let node = forest.node_mut(*nid);
      node.value.add_sample(values[ix], 1);
      bounds[ix].update_bounds(values[ix]);
    }
    //print!("values: {values:?} agents in backprop: ");
    for (nids, (agent, action)) in trajectory.branch.iter().rev() {
      //print!(" {agent}");
      for ix in 0..nids.len() {
        let node = forest.node_mut(nids[ix].0);
        if ix == *agent {
          //print!(" up {}", nids[ix].1);
          let data = node.actions.get_mut(action).unwrap();
          data.action_reward.add_sample(nids[ix].1, 1);
          data.value_of_next_state.add_sample(values[ix], 1);
        }

        values[ix] += nids[ix].1;
        node.value.add_sample(values[ix], 1);
        bounds[ix].update_bounds(values[ix]);
      }
    }
    //println!();
  }

  pub fn create_workers(&self, count: usize) -> Vec<Worker<P::HiddenState, P::Action>> {
    let guard = self.forest.write().unwrap();
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
      result.push(Worker {
        states_in_flight: self
          .problem
          .sample_h_state_batched(&self.b_state, self.block_size as usize),
        trajectories_in_flight: vec![self.empty_trajectory(&guard); self.block_size as usize],
        trajectories_awaiting_expansion: vec![],
        states_awaiting_expansion: vec![],
        trajectories_awaiting_backprop: vec![],
      });
    }
    result
  }

  pub fn get_policy(&self) -> Vec<(P::Action, f32, f32)> {
    let guard = self.forest.read().unwrap();
    let agent_ix = self
      .problem
      .agent_to_act(&self.problem.sample_h_state(&self.b_state))
      .into() as usize;
    let root_id = guard.roots()[agent_ix];
    let root = guard.node(root_id);
    let root_s_count = root.select_count() as f32;
    root
      .actions
      .iter()
      .map(|(a, data)| {
        (
          a.clone(),
          data.select_count() as f32 / root_s_count,
          data.value(),
        )
      })
      .collect()
  }

  fn empty_trajectory(
    &self,
    forest_g: &Forest<P::Action, P::Observation>,
  ) -> Trajectory<P::Action> {
    Trajectory {
      current_: forest_g.roots(),
      branch: vec![],
    }
  }

  fn restart_trajectory(
    &self,
    forest_g: &Forest<P::Action, P::Observation>,
    trajectory: &mut Trajectory<P::Action>,
  ) {
    trajectory.current_ = forest_g.roots();
    trajectory.branch = vec![];
  }
}

impl<A: Clone + Debug> Debug for Trajectory<A> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self.current_)
  }
}
