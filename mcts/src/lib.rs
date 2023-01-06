use std::cell::Cell;



struct Bounds {

}


struct Average {
  stats: Cell<(f32, u32)>,
}

impl Average {
  fn new() -> Self {
    Average { stats: Cell::new((0.0, 0)) }
  }

  fn mean(&self) -> f32 {self.stats.get().0}

  fn sample_count(&self) -> u32 {
    self.stats.get().1
  }

  fn add_sample(&self, v: f32, n: u32) {
    let (old_s, old_n )= self.stats.get();
    let new_n = old_n + n;
    let new_s = old_s + (v - old_s) / new_n as f32;
    self.stats.replace((new_s, new_n));
  }
}
