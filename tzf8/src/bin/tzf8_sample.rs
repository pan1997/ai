use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use mcts::SearchLimit;
use ml::accumulate_rewards;
use ml::playout;
use tzf8::Tzf8;
use lib::MctsProblem;
use mcts::bandits::Uct;
use mcts::rollout::RandomRollout;


fn main() {
  let m = Arc::new(Tzf8);
  let counts = Arc::new(Mutex::new(BTreeMap::new()));

  crossbeam::scope(|s| {
    let handles: Vec<_> = (0..100).map(|_| {
      let h = s.spawn(|_| {
        let mut start = m.start_state();
        let limit = SearchLimit::new(10000);
        let bandit_policy = Uct(1.2);
        let t = playout(m.clone(), &mut start, 1, limit, bandit_policy, u32::MAX, RandomRollout(40), true);
        let r = accumulate_rewards(&Tzf8, &t);
        let largest_tile = start.largest_tile();
        println!("{}, {}", r[0], largest_tile);
        (r, largest_tile)
      });
      h
    }).collect();

    for handle in handles {
      let (_, largest_tile) = handle.join().unwrap();
      let mut cnts = counts.lock().unwrap();
      let old = *cnts.get(&largest_tile).unwrap_or(&0);
      cnts.insert(largest_tile, old + 1);
    }
  }).unwrap();
  println!("counts: {counts:?}");
}