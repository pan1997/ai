use std::sync::{Condvar, Mutex, RwLock};

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

/*
struct Batch<A, B, C> {
  size: usize,
  computation: C,
  
  input: Mutex<Vec<A>>,
  output: RwLock<Option<Vec<B>>>,
  full: Condvar
}

impl<A, B, C> Batch<A, B, C> {
  fn new(size: usize, computation: C) -> Self {
    Self {
      size,
      computation,
      input: Mutex::new(Vec::with_capacity(N)),
      output: RwLock::new(None),
      full: Condvar::new()
    }
  }

  fn process(&self, x: A) -> B where C: Fn(&[A]) -> [B] {
    let mut input_guard = self.input.lock().unwrap();
    // clear if full?
    let index = input_guard.len();
    input_guard.push(x);

    if input_guard.len() >= self.size {
      let mut output_guard = self.output.write().unwrap();
      output_guard.replace((self.computation)(&input_guard));
      self.full.notify_all();
    } else {
      let _guard = self.full.wait_while(input_guard, |g| g.len() < self.size).unwrap();
    }
    self.output.read().unwrap()

    unimplemented!()
  }
}*/