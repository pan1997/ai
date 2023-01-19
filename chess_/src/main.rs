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
  let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();
  let wc = 20;
  //println!("created");

  let join_hanldes = (0..wc).map(|_| {
    let search_local = search.clone();
    rt.spawn(async move {
      let mut w = search_local.create_workers(1).await;
      search_local.start(&mut w[0]).await
    })
  });

  rt.block_on(async move {
    for handle in join_hanldes {
      handle.await;
    }
  });

  let forest = search.forest.blocking_read();
  //println!("{:?}", forest);
  //save(&forest, File::create("chess.dot").unwrap(), 500, 10);
}

fn main() {
  bench3();
}
