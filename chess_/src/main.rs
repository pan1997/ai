mod wrap;
mod wrapv2;

use std::{fs::File, time::Duration};

use lib::mmdp::{State, FODMMDP};
use lib_v2::FullyObservableDeterministicMctsProblem;
use mcts::{
  time_manager::Limit,
  tree::{render::save_tree, Node},
  util::{EmptyExpansion, UctTreePolicy},
  Search,
};
use mcts_v2::SearchLimit;
use mcts_v2::search::{Search as Searchv2};
use mcts_v2::bandits::UctBandit;
use mcts_v2::forest::render::save;
use wrapv2::Game2;

use crate::wrap::Game;

fn bench1() {
  let args: Vec<String> = std::env::args().collect();

  let g = Game {};
  let state = g.start_state();
  let t_black = Node::new(&[]);
  let t_white = Node::new(&state.legal_actions());

  let s = Search::new(&g, UctTreePolicy(2.5), EmptyExpansion, 80, true);
  let count: u32 = args[1].parse().unwrap();
  for _ in 0..count {
    s.one_block(&mut state.clone(), vec![&t_black, &t_white]);
  }
  //save_tree(&t_white, File::create("white.dot").unwrap(), 20, 10);
}

fn bench2() {
  let g = Game {};
  let state = g.start_state();
  let t_black = Node::new(&[]);
  let t_white = Node::new(&state.legal_actions());

  let s = Search::new(&g, UctTreePolicy(2.5), EmptyExpansion, 80, false);
  let limit = Limit::time(Duration::from_secs(2), 1024);
  limit.start(&s, &state, vec![&t_black, &t_white]);
  //save_tree(&t_white, File::create("white.dot").unwrap(), 200, 10);
}

fn bench3() {
  let args: Vec<String> = std::env::args().collect();
  let count: u32 = args.get(1).map(|arg| arg.parse().unwrap()).unwrap();
  let g = Game2 {};
  let state = g.start_state();
  let limit = SearchLimit::new(count);
  let search = Searchv2::new(
    &g, 
    &state, 1, limit, UctBandit(2.5)
  );
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
