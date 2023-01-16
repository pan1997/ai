use std::collections::{VecDeque, BTreeSet};

use lib::sat::CNF;

type Literal = i64;
type Variable = i64;


struct DpllState<'a> {
  cnf: &'a CNF,

  stack: Vec<DpllAction>,
  
  clause_infos: Vec<ClauseInfo>,
  variable_infos: Vec<VariableInfo>,

  unprocessed_assignments: VecDeque<Literal>,

  // todo: should this be merged with variable info and
  // changed to something like priority queue?
  unassigned_variables: BTreeSet<Variable> 
}

enum DpllAction {
  Assign(Literal),

}

struct VariableInfo {
  positive_clauses: Vec<usize>,
  negative_clauses: Vec<usize>,
}


struct ClauseInfo {
  // the number of literals remaining un assigned in this clause
  remaining_literal_count: u32,


  // if set true, this denotes the assignement that set this clause true
  solved_by: Option<Literal>,
}

// return false if the operation failed
fn assign<'a>(state: &mut DpllState<'a>, literal: Literal) -> bool {
  let var = literal.abs();
  if !state.unassigned_variables.contains(&var) {
    false
  }
  else {
    state.stack.push(DpllAction::Assign(literal));
    state.unassigned_variables.remove(&var);
    // add unit propogate
    let var_info = &state.variable_infos[var as usize];
    let negated = literal < 0;

    

    true
  }
}


fn unit_propogate<'a>(state: &mut DpllState<'a>, literal: Literal) -> bool {
  true
}


#[cfg(test)]
mod tests {
  use super::*;

}
