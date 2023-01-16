use std::fmt::Display;

use lib::MctsProblem;
use rand::seq::SliceRandom;

use crate::{forest::Node, NodeInit};

#[derive(Copy, Clone)]
pub struct RandomRollout(pub u32);

#[derive(Clone, Copy)]
pub struct EmptyInit;

impl<P: MctsProblem> NodeInit<P> for RandomRollout where P::HiddenState: Clone, P::Action: Display {
  fn init_node(
    &self,
    problem: &P,
    state: &<P as MctsProblem>::HiddenState,
    node_of_current_agent: &mut crate::forest::Node<
      <P as MctsProblem>::Action,
      <P as MctsProblem>::Observation,
    >,
  ) -> Vec<f32> {
    // static policy scores are equal for all actions
    let ac = node_of_current_agent.actions.len();
    if ac > 0 {
      let v = 1.0 / ac as f32;
      for (_, data) in node_of_current_agent.actions.iter_mut() {
        data.static_policy_score = v;
      }
    }

    let mut _state = state.clone();
    let mut total = vec![0.0; problem.agents().len()];
    let mut factor = 1.0;
    let mut horizon = self.0;
    while !problem.check_terminal(&_state) && horizon > 0 {
      let actions = problem.legal_actions(&_state);

      //if actions.is_empty() {
      // sanity check in case the check terminal doesn't work correclty
      //  break;
      //}

      let random_action = actions.choose(&mut rand::thread_rng()).unwrap();
      //print!("{random_action} ");
      let ro = problem.apply_action(&mut _state, random_action);
      for ix in 0..total.len() {
        total[ix] += factor * ro[ix].0;
      }
      factor *= problem.discount();
      horizon -= 1;
    }
    //println!("horizon_remaining:{horizon}, values: {total:?}, factor: {factor}");

    total
  }
}

impl<P: MctsProblem> NodeInit<P> for EmptyInit {
  fn init_node(
    &self,
    problem: &P,
    _state: &<P as MctsProblem>::HiddenState,
    node_of_current_agent: &mut Node<<P as MctsProblem>::Action, <P as MctsProblem>::Observation>,
  ) -> Vec<f32> {
    let ac = node_of_current_agent.actions.len();
    if ac > 0 {
      let v = 1.0 / ac as f32;
      for (_, data) in node_of_current_agent.actions.iter_mut() {
        data.static_policy_score = v;
      }
    }
    vec![0.0; problem.agents().len()]
  }
}
