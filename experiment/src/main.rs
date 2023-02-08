
use std::sync::mpsc;


fn main() {
  let (rx, tx) = mpsc::channel();
  rx.send(5).unwrap();
  let x = tx.recv().unwrap();
  println!("x: {x}");
}


