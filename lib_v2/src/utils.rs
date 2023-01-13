#[derive(Debug)]
pub struct RunningAverage {
  mean: f32,
  count: u32,
}

impl RunningAverage {
  pub fn new() -> Self {
    Self {
      mean: 0.0,
      count: 0,
    }
  }

  pub fn value(&self) -> f32 {
    self.mean
  }
}

#[derive(Debug, Clone)]
pub struct Bounds {
  low: f32,
  high: f32,
}

impl Bounds {
  pub fn new_known(low: f32, high: f32) -> Self {
    Bounds { low, high }
  }

  pub fn new() -> Self {
    Bounds {
      low: f32::MAX,
      high: f32::MIN,
    }
  }

  pub fn normalise(&self, v: f32) -> f32 {
    (v - self.low) / (self.high - self.low)
  }

  pub fn update_bounds(&mut self, v: f32) {
    if v < self.low {
      self.low = v;
    }
    if v > self.high {
      self.high = v
    }
  }
}
