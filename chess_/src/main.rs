mod wrap;

use lib::FullyObservableDeterministicMctsProblem;
use mcts::{bandits::Puct, search::Search as Searchv2, SearchLimit, EmptyInit};
use wrap::Game;

fn bench3() {
  let args: Vec<String> = std::env::args().collect();
  let count: u32 = args.get(1).map(|arg| arg.parse().unwrap()).unwrap();
  let g = Game {};
  let state = g.start_state();
  let limit = SearchLimit::new(count);
  let search = Searchv2::new(&g, &state, 1, limit, Puct(2.5), EmptyInit);
  let rt = tokio::runtime::Builder::new_current_thread()
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
