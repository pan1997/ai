mod wrap;

use std::{fs::File, time::Duration};

use lib::mmdp::{State, FODMMDP};
use lib_v2::FullyObservableDeterministicMctsProblem;
use mcts::{bandits::UctBandit, forest::render::save, search::Search as Searchv2, SearchLimit, EmptyInit};
use wrap::Game2;

fn bench3() {
  let args: Vec<String> = std::env::args().collect();
  let count: u32 = args.get(1).map(|arg| arg.parse().unwrap()).unwrap();
  let g = Game2 {};
  let state = g.start_state();
  let limit = SearchLimit::new(count);
  let search = Searchv2::new(&g, &state, 1, limit, UctBandit(2.5), EmptyInit);
  let mut rt = tokio::runtime::Builder::new_current_thread()
    .build()
    .unwrap();
  let mut worker = rt.block_on(search.create_workers(1));
  println!("created");
  rt.block_on(search.start(&mut worker[0]));
  //let forest = search.forest.blocking_read();
  //println!("{:?}", forest);
  //save(&forest, File::create("chess.dot").unwrap(), 200, 10);
}

fn main() {
  bench3();
}
