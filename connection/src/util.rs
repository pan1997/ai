use std::ops::Index;

use fixedbitset::FixedBitSet;



#[derive(Clone)]
pub struct RectBitSet<const H: usize, const W: usize> {
  internal: FixedBitSet
}


impl<const H: usize, const W: usize> RectBitSet<H, W> {
  pub fn new() -> Self {
    Self {
      internal: FixedBitSet::with_capacity(H * W)
    }
  }

  fn flatten(&self, index: (usize, usize)) -> usize {
    index.0 * W + index.1
  }

  fn flatten_i(&self, index: (i32, i32)) -> usize {
    index.0 as usize * W + index.1 as usize 
  }

  pub fn ray_count(&self, mut start: (i32, i32), delta: (i32, i32), bound: u8) -> u8 {
    for count in 1..(bound + 1) {
      start.0 += delta.0;
      start.1 += delta.1;
      if start.0 < 0 || start.0 > H as i32 || start.1 < 0 || start.1 > W as i32 || !self.internal[self.flatten_i(start)] {
        return count - 1;
      }
    }
    bound
  }

  pub fn set(&mut self, index: (usize, usize), value: bool) {
    self.internal.set(self.flatten(index), value)
  }
}

impl<const H: usize, const W: usize> Index<(usize, usize)> for RectBitSet<H, W> {
  type Output = bool;
  fn index(&self, index: (usize, usize)) -> &bool {
    self.internal.index(self.flatten(index))
  }
}

