use std::fmt::Display;

use lib::MctsProblem;
use rand::seq::SliceRandom;

use crate::{forest::Node, Expansion};

#[derive(Copy, Clone)]
pub struct RandomRollout(pub u32);

#[derive(Clone, Copy)]
pub struct EmptyInit;

impl<P: MctsProblem> Expansion<P> for RandomRollout
where
  P::HiddenState: Clone,
  P::Action: Display,
{
  fn expand(
    &self,
    problem: &P,
    state: &<P as MctsProblem>::HiddenState,
  ) -> (Vec<f32>, Vec<(<P as MctsProblem>::Action, f32)>) {
    let mut _state = state.clone();
    let mut total = vec![0.0; problem.agents().len()];
    let mut factor = 1.0;
    let mut horizon = self.0;
    while !problem.check_terminal(&_state) && horizon > 0 {
      let actions = problem.legal_actions(&_state);
      let random_action = actions.choose(&mut rand::thread_rng()).unwrap();
      //print!("{random_action} ");
      let ro = problem.apply_action(&mut _state, random_action);
      for ix in 0..total.len() {
        total[ix] += factor * ro[ix].0;
      }
      factor *= problem.discount();
      horizon -= 1;
    }
    (total, vec![])
  }
}

impl<P: MctsProblem> Expansion<P> for EmptyInit {
  fn expand(
    &self,
    p: &P,
    s: &<P as MctsProblem>::HiddenState,
  ) -> (Vec<f32>, Vec<(<P as MctsProblem>::Action, f32)>) {
    let ac = p.agents().len();
    (vec![0.0; ac], vec![])
  }
}
