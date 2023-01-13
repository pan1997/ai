use std::{fmt::Debug, ops::Deref};

use lib_v2::{utils::Bounds, MctsProblem};
use tokio::sync::RwLock;

use crate::{
  bandits::Bandit,
  forest::{Forest, Node, NodeId},
  SearchLimit,
};

pub struct Search<'a, 'b, P: MctsProblem, B> {
  problem: &'a P,
  b_state: &'b P::BeliefState,
  forest: RwLock<Forest<P::Action, P::Observation>>,
  block_size: u32,
  limit: SearchLimit,
  bandit_policy: B,
  score_bounds: Vec<Bounds>,
}

#[derive(Clone)]
pub struct Worker<S, A: Clone> {
  states_in_flight: Vec<S>,
  trajectories_in_flight: Vec<Trajectory<A>>,

  states_awaiting_expansion: Vec<S>,
  trajectories_awaiting_expansion: Vec<Trajectory<A>>,
}

#[derive(Clone)]
struct Trajectory<A: Clone> {
  // one node in each player's tree
  current_: Vec<NodeId>,

  // nodeId, emitted reward and the selected action
  branch: Vec<(Vec<(NodeId, f32)>, A)>,
}

impl<'a, 'b, P: MctsProblem, B> Search<'a, 'b, P, B>
where
  B: Bandit<P::HiddenState, P::Action, P::Observation>,
  // todo: remove this requirement
  P::HiddenState: Clone,
{
  pub fn new(
    problem: &'a P,
    b_state: &'b P::BeliefState,
    block_size: u32,
    limit: SearchLimit,
    bandit_policy: B,
  ) -> Self {
    let mut forest = Forest::new(800);
    let agent_count = problem.agents().len();
    for _ in 0..agent_count {
      forest.new_root();
    }

    // todo: see if this can be removed
    let h_state = problem.sample_h_state(b_state);
    let current_agent_ix = problem.agent_to_act(&h_state).into() as usize;
    let node = forest.node_mut(forest.roots()[current_agent_ix]);
    node.create_actions(problem.legal_actions(&h_state));
    // todo remove this hack to avoid re expansion
    forest.roots().into_iter().for_each(|id| forest.node_mut(id).increment_select_count());

    Search {
      problem,
      b_state,
      forest: RwLock::new(forest),
      block_size,
      limit,
      bandit_policy,
      score_bounds: vec![Bounds::new_known(-10.0, 10.0); agent_count],
    }
  }

  pub async fn start(&self, worker: &mut Worker<P::HiddenState, P::Action>) {
    loop {
      // select actions
      let agents_and_actions: Vec<_> = {
        let guard = self.forest.read().await;
        // check if search budget remains
        let select_count_root = guard.node(guard.roots()[0]).select_count();
        if !self.limit.more(select_count_root) {
          return;
        }
        worker
          .trajectories_in_flight
          .iter_mut()
          .zip(worker.states_in_flight.iter_mut())
          .map(|(trajectory, state)| {
            // todo: relax assumption that starting state is non terminal
            if self.problem.check_terminal(&state) {
              // state is terminal
              self.backpropogate(trajectory, vec![0.0]);
              // todo add batching support
              *state = self.problem.sample_h_state(self.b_state);
              self.restart_trajectory(&guard, trajectory);
            }
            // its guaranteed that the state is not terminal
            let current_agent = self.problem.agent_to_act(state);
            let current_agent_ix = current_agent.into() as usize;

            // this is a new node that has never been selected till now
            // it has to be expanded
            if guard
              .node(trajectory.current_[current_agent_ix])
              .select_count()
              == 0
            {
              // push this state trajectory pair to the expansion queue (processed later)
              worker.states_awaiting_expansion.push(state.clone());
              worker
                .trajectories_awaiting_expansion
                .push(trajectory.clone());

              // todo: expansion
              if worker.states_awaiting_expansion.len() >= self.block_size as usize {

              }

              *state = self.problem.sample_h_state(self.b_state);
              self.restart_trajectory(&guard, trajectory);
            }

            // its guaranteed that the trajectory is not terminal
            (
              current_agent_ix,
              self.bandit_policy.select(
                state,
                guard.node(trajectory.current_[current_agent_ix]),
                &self.score_bounds[current_agent_ix],
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

      // descend tree
      {
        let mut guard = self.forest.write().await;
        worker
          .trajectories_in_flight
          .iter_mut()
          .zip(worker.states_in_flight.iter_mut())
          .zip(
            actions
              .into_iter()
              .zip(agents.into_iter())
              .zip(outcomes.into_iter()),
          )
          .for_each(|((trajectory, state), ((action, agent_ix), outcomes_and_rewards))| {
            // increment select_counts
            // descend nodes

            // todo: capacity
            let mut children_ix = vec![];
            let mut branch_entry = vec![];
            for (ix, node_id) in trajectory.current_.iter().enumerate() {
              {
                let mut node = guard.node_mut(*node_id);
                node.increment_select_count();
                // on the first selection, we need to create the actions map
                // the policy weights are updated when the node is expanded
                // till then, the nodes are descended randomly
                if ix == agent_ix && node.select_count() == 1 {
                  node.create_actions(self.problem.legal_actions(state));
                }

                node
                  .actions
                  .get_mut(&action)
                  .unwrap()
                  .increment_select_count();
              }
              children_ix.push(guard.get_id_of_child(*node_id, &outcomes_and_rewards[ix].1));
              branch_entry.push((*node_id, outcomes_and_rewards[ix].0));
            }
            trajectory.current_ = children_ix;
            trajectory.branch.push((branch_entry, action));
          });
      }
    }
  }

  fn backpropogate(&self, trajectory: &Trajectory<P::Action>, values: Vec<f32>) {}

  pub async fn create_workers(&self, count: usize) -> Vec<Worker<P::HiddenState, P::Action>> {
    let guard = self.forest.write().await;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
      result.push(Worker {
        states_in_flight: self
          .problem
          .sample_h_state_batched(self.b_state, self.block_size as usize),
        trajectories_in_flight: vec![self.empty_trajectory(&guard); self.block_size as usize],
        trajectories_awaiting_expansion: vec![],
        states_awaiting_expansion: vec![],
      });
    }
    result
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

impl<'a, 'b, P: MctsProblem, B> Debug for Search<'a, 'b, P, B>
where
  P::Observation: Debug,
  P::Action: Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self.forest.blocking_read())
  }
}
