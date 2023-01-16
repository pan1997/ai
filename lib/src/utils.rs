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

  pub fn count(&self) -> u32 {
    self.count
  }

  pub fn add_sample(&mut self, v: f32, c: u32) {
    let new_c = c + self.count;
    self.mean += (v - self.mean) * (c as f32) / (new_c as f32);
    self.count = new_c;
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
      low: 1.0,
      high: 0.0,
    }
  }

  pub fn normalise(&self, v: f32) -> f32 {
    if self.low >= self.high {
      0.0
    } else {
      (v - self.low) / (self.high - self.low)
    }
  }

  pub fn update_bounds(&mut self, v: f32) {
    if self.low > self.high {
      self.low = v;
      self.high = v;
    } else {
      if v < self.low {
        self.low = v;
      }
      if v > self.high {
        self.high = v
      }
    }
  }
}
