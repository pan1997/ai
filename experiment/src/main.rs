
use std::sync::{Condvar, Mutex, RwLock};

struct BatchCollecter<C: Fn(&Vec<u32>) -> Vec<u32> > {
  size: usize,
  input: Mutex<Vec<u32>>,
  output: RwLock<Option<Vec<u32>>>,
  //barrier: Barrier
  full: Condvar,
  computation: C
}

impl<C: Fn(&Vec<u32>) -> Vec<u32>> BatchCollecter<C> {
  fn new(
    size: usize,
    computation: C
  ) -> Self {
    Self { 
      size, 
      input: Mutex::new(Vec::new()), 
      output: RwLock::new(None), 
      full: Condvar::new(), 
      computation 
    }
  }

  fn submit(&self, x: u32) -> u32 {
    let mut guard = self.input.lock().unwrap();
    // clear if full
    
    if guard.len() >= self.size {
      guard.clear();
      self.output.write().unwrap().take();
    }

    let index = guard.len();
    guard.push(x);

    if guard.len() >= self.size {
      let mut w = self.output.write().unwrap();
      println!("processing: {:?}", guard);
      w.replace((self.computation)(&guard));
      self.full.notify_all();
    } else {
      let _guard = self.full.wait_while(guard, |g| g.len() < self.size).unwrap();
    }
    self.output.read().unwrap().as_ref().unwrap()[index]
  }

  fn done(&self) -> bool {
    self.output.read().unwrap().is_some()
  }

  fn clear(&self) {
    self.input.lock().unwrap().clear();
    self.output.write().unwrap().take();
  }
}



fn main() {
  let b = BatchCollecter::new(10, |batch| batch.iter().map(|x| *x).collect());
  rayon::scope(|s| {
    for ix in 0..40 {
      let m = ix;
      let batch = &b;
      s.spawn( move |_|  {
        println!("ix: {m}");
        let mut result = batch.submit(m);
        println!("ix: {m} result: {result}");
      });
    }
  });
}


