use std::fmt::Display;

use lib::{BeliefState, State, MPOMDP};

use crate::{tree::Node, Search, TreeExpansion, TreeExpansionBlock, TreePolicy};

impl<'a, P, T, E> Search<'a, P, T, E>
where
  P: MPOMDP,
  T: TreePolicy<P::State>,
{
  // one tree for each agent
  fn sample<'b>(
    &self,
    state: &mut P::State,
    mut trees: Vec<&'b Node<P::Action, P::Observation>>,
  ) -> Vec<SelectStep<'b, P::Agent, P::Action, P::Observation>> {
    let mut result = vec![];

    for _ in 0..self.horizon {
      // increment select count
      for tree in trees.iter() {
        tree.increment_select_count();
      }

      if state.is_terminal() {
        result.push(SelectStep {
          nodes: trees,
          next: SelectStepNext::Terminal,
        });
        return result;
      }

      let current_agent = state.current_agent().unwrap();
      let current_agent_index: usize = current_agent.into();

      let selected_action = self.tree_policy.select_action(
        &state,
        &trees[current_agent_index],
        &self.bounds[current_agent_index],
        true,
      );

      let rewards_and_observations = state.apply_action(selected_action);

      if self.expand_unseen
        && trees[current_agent_index]
          .next_node(&rewards_and_observations[current_agent_index].1)
          .is_none()
      {
        result.push(SelectStep {
          nodes: trees,
          next: SelectStepNext::ToExpand {
            rewards_and_observations,
          },
        });
        return result;
      }

      let mut next_trees = Vec::with_capacity(trees.len());
      for (ix, tree) in trees.iter().enumerate() {
        let next_node = {
          let n = tree.next_node(&rewards_and_observations[ix].1);
          if n.is_none() {
            let actions = if self.expand_unseen || ix != state.current_agent().unwrap().into() {
              vec![]
            } else {
              state.legal_actions()
            };
            tree.create_new_node(rewards_and_observations[ix].1.clone(), actions);
            tree.next_node(&rewards_and_observations[ix].1).unwrap()
          } else {
            n.unwrap()
          }
        };
        next_trees.push(next_node);
      }

      result.push(SelectStep {
        nodes: trees,
        next: SelectStepNext::Next {
          agent: current_agent,
          action: selected_action.clone(),
          rewards_and_observations,
        },
      });
      trees = next_trees;
    }
    // reach here on reaching horizon
    result.push(SelectStep {
      nodes: trees,
      next: SelectStepNext::Terminal,
    });
    result
  }

  /*
  fn select_step<'b>(
    &self,
    state: &mut P::State,
    nodes: Vec<&'b Node<P::Action, P::Observation>>
  ) -> SelectStep<'b, P::Agent, P::Action, P::Observation> {
    // 1. increment select count
    for node in nodes.iter() {
      node.increment_select_count();
    }

    // todo: check if termination info can be stored in trees

    // 2. terminate if state is terminal
    if state.is_terminal() {
      return SelectStep {
        nodes,
        next: SelectStepNext::Terminal,
      }
    }

    let current_agent = state.current_agent().unwrap();
    let current_agent_index: usize = current_agent.into();
    // 3. select action using tree policy
    let selected_action = self.tree_policy.select_action(
      &state,
      &nodes[current_agent_index],
      &self.bounds[current_agent_index],
      true,
    );

    // 4. apply action
    let rewards_and_observations = state.apply_action(selected_action);

    // 5. expand if configured to terminate
    if self.expand_unseen
        && nodes[current_agent_index]
          .next_node(&rewards_and_observations[current_agent_index].1)
          .is_none()
      {
        return SelectStep {
          nodes,
          next: SelectStepNext::ToExpand {
            rewards_and_observations,
          },
        }
      }

      // 6. descend if needed
      let mut next_trees = Vec::with_capacity(nodes.len());
      for (ix, tree) in nodes.iter().enumerate() {
        let next_node = {
          let n = tree.next_node(&rewards_and_observations[ix].1);
          if n.is_none() {
            let actions = if self.expand_unseen || ix != state.current_agent().unwrap().into() {
              vec![]
            } else {
              state.legal_actions()
            };
            tree.create_new_node(rewards_and_observations[ix].1.clone(), actions);
            tree.next_node(&rewards_and_observations[ix].1).unwrap()
          } else {
            n.unwrap()
          }
        };
        next_trees.push(next_node);
      }
    unimplemented!()
  }*/

  fn propagate<'b>(
    &self,
    trajectory: &Vec<SelectStep<'b, P::Agent, P::Action, P::Observation>>,
    mut terminal_value: Vec<f32>,
  ) -> Vec<f32> {
    for step in trajectory.into_iter().rev() {
      if let SelectStepNext::Next {
        agent,
        action,
        rewards_and_observations,
      } = &step.next
      {
        let agent_ix: usize = <P::Agent as Into<usize>>::into(*agent);
        let a = &step.nodes[agent_ix].actions[&action];
        a.value_of_next_state
          .add_sample(terminal_value[agent_ix], 1);
        a.action_reward
          .add_sample(rewards_and_observations[agent_ix].0, 1);
        for ix in 0..terminal_value.len() {
          terminal_value[ix] += rewards_and_observations[ix].0;
        }
      }

      for (agent_ix, node) in step.nodes.iter().enumerate() {
        node.value.add_sample(terminal_value[agent_ix], 1);
      }
    }
    terminal_value
  }

  pub fn initialize(&self, state: &P::State) -> Vec<Node<P::Action, P::Observation>> {
    let a = self.problem.all_agents().len();
    let current_agent_index: usize = state.current_agent().unwrap().into();
    (0..a)
      .map(|ix| {
        if ix == current_agent_index {
          Node::new(&state.legal_actions())
        } else {
          Node::new(&[])
        }
      })
      .collect()
  }
}

impl<'a, P, T, E> Search<'a, P, T, E>
where
  P: MPOMDP,
  T: TreePolicy<P::State>,
  E: TreeExpansion<P::State>,
{
  pub fn once<'b>(&self, state: &mut P::State, trees: Vec<&'b Node<P::Action, P::Observation>>) {
    let p = trees.len();
    let trajectory = self.sample(state, trees);
    trajectory.last().map(|step| {
      let trajectory_value = if let SelectStepNext::ToExpand {
        rewards_and_observations,
      } = &step.next
      {
        self.tree_expansion.create_node_and_estimate_value(
          &step.nodes,
          &rewards_and_observations,
          &state,
        )
      } else {
        vec![0.0; p]
      };
      self.propagate(&trajectory, trajectory_value);
    });
  }
}

impl<'a, P, T, E> Search<'a, P, T, E>
where
  P: MPOMDP,
  T: TreePolicy<P::State>,
  E: TreeExpansionBlock<P::State>,
{
  pub fn one_block<'b>(
    &self,
    belief_state: &P::BeliefState,
    trees: Vec<&'b Node<P::Action, P::Observation>>,
  ) {
    let p = trees.len();
    let mut trajectories = Vec::with_capacity(self.block_size as usize);
    let mut nodes = Vec::with_capacity(self.block_size as usize);
    let mut last_rewards_and_observations = Vec::with_capacity(self.block_size as usize);
    let mut states = Vec::with_capacity(self.block_size as usize);
    for _ in 0..self.block_size {
      let mut state = belief_state.sample_state();
      let trajectory = self.sample(&mut state, trees.clone());

      let process = trajectory
        .last()
        .map(|step| {
          if let SelectStepNext::ToExpand {
            rewards_and_observations,
          } = &step.next
          {
            nodes.push(step.nodes.clone());
            last_rewards_and_observations.push(rewards_and_observations.clone());
            true
          } else {
            self.propagate(&trajectory, vec![0.0; p]);
            false
          }
        })
        .unwrap_or(false);
      if process {
        trajectories.push(trajectory);
        states.push(state);
      }
    }

    let expansion_result = self.tree_expansion.create_nodes_and_estimate_values(
      &nodes,
      &last_rewards_and_observations,
      &states,
    );

    for ix in 0..trajectories.len() {
      self.propagate(&trajectories[ix], expansion_result[ix].clone());
    }
  }
}

struct SelectStep<'a, Agent, Action, Observation> {
  nodes: Vec<&'a Node<Action, Observation>>,
  next: SelectStepNext<Agent, Action, Observation>,
}

enum SelectStepNext<Agent, Action, Observation> {
  ToExpand {
    rewards_and_observations: Vec<(f32, Observation)>,
  },
  Terminal,
  Next {
    agent: Agent,
    action: Action,
    rewards_and_observations: Vec<(f32, Observation)>,
  },
}

impl<A, Aa: Display, O: Display> Display for SelectStepNext<A, Aa, O> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      SelectStepNext::Terminal => write!(f, "Terminal"),
      SelectStepNext::ToExpand {
        rewards_and_observations: _,
      } => write!(f, "ToExpand()"),
      SelectStepNext::Next {
        agent: _,
        action,
        rewards_and_observations: _,
      } => write!(f, "Next({})", action),
    }
  }
}

impl<Agent, Action, Observation> SelectStepNext<Agent, Action, Observation> {
  fn terminal(&self) -> bool {
    match self {
      SelectStepNext::ToExpand {
        rewards_and_observations,
      } => false,
      _ => true,
    }
  }
}
