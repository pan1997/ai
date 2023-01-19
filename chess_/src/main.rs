mod wrap;
use std::{fs::File, sync::Arc};

use lib::FullyObservableDeterministicMctsProblem;
use mcts::{
  bandits::{Puct, Uct},
  forest::render::save,
  rollout::RandomRollout,
  search::Search as Searchv2,
  SearchLimit,
};
use wrap::Game;

fn bench3() {
  let args: Vec<String> = std::env::args().collect();
  let count: u32 = args.get(1).map(|arg| arg.parse().unwrap()).unwrap();
  let g = Arc::new(Game {});
  let state = Arc::new(g.start_state());
  let limit = SearchLimit::new(count);
  let search = Arc::new(Searchv2::new(
    g,
    state,
    1,
    limit,
    Uct(2.5),
    RandomRollout(120),
  ));
  let wc = 12;

  crossbeam::scope(|s| {
    for _ in 0..wc {
      s.spawn(|_| {
        let mut worker = search.create_workers(1);
        search.start(&mut worker[0]);
      });
    }
  }).unwrap();

  let forest = search.forest.read().unwrap();
  println!("{:?}", forest);
  save(&forest, File::create("chess.dot").unwrap(), 5000, 10);
}

fn main() {
  bench3();
}
