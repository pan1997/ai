



pub struct CNF {
  pub var_count: usize,
  // variables are numbered 1, 2, 3...
  // negative signify negated var
  pub clauses: Vec<Vec<i64>>
}


impl CNF {
  pub fn new(vc: usize) -> Self{
    Self { var_count: vc, clauses: vec![] }
  }

  pub fn add_clause(&mut self, clause: Vec<i64>) {
    for c in clause.iter() {
      if *c > self.var_count as i64 {
        panic!("unknown variable{c}")
      }
    }
    self.clauses.push(clause);
  }
}