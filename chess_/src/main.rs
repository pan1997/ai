mod wrap;

use std::fs::File;

use lib::mmdp::{State, FODMMDP};
use mcts::{
  tree::{render::save_tree, Node},
  util::{EmptyExpansion, UctTreePolicy},
  Search,
};

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
  save_tree(&t_white, File::create("white.dot").unwrap(), 20, 10);
}

fn main() {
  bench1();
}
