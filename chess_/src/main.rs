mod wrap;
use std::fs::File;

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
  let g = Game {};
  let state = g.start_state();
  let limit = SearchLimit::new(count);
  let search = Searchv2::new(&g, &state, 1, limit, Uct(2.5), RandomRollout(120));
  let rt = tokio::runtime::Builder::new_multi_thread()
    .build()
    .unwrap();
  let mut worker = rt.block_on(search.create_workers(1));
  //println!("created");

  rt.block_on(search.start(&mut worker[0]));
  let forest = search.forest.blocking_read();
  //println!("{:?}", forest);
  //save(&forest, File::create("chess.dot").unwrap(), 500, 10);
}

fn main() {
  bench3();
}
