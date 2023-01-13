pub mod bandits;
pub mod forest;
pub mod search;

pub struct SearchLimit {
  node_count: Option<u32>,
}

impl SearchLimit {
  fn more(&self, n: u32) -> bool {
    if self.node_count.map(|l| n > l).unwrap_or(false) {
      return false;
    }
    // more checks

    true
  }

  pub fn new(n: u32) -> Self {
    SearchLimit {
      node_count: Some(n),
    }
  }
}
