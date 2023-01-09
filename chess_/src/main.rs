mod wrap;
use std::env::Args;

use lib::mmdp::{State, FODMMDP};
use mcts::{
  tree::{render::save_tree, Node},
  util::{EmptyExpansion, UctTreePolicy},
  Search,
};

use crate::wrap::Game;

fn main() {
  //let args: Vec<String> = std::env::args().collect();

  let g = Game {};
  let state = g.start_state();
  let t_black = Node::new(&[]);
  let t_white = Node::new(&state.legal_actions());

  let s = Search::new(&g, UctTreePolicy(2.5), EmptyExpansion, 80);
  //let count: u32 = args[1].parse().unwrap();
  let count = 30000;
  for _ in 0..count {
    s.once(&mut state.clone(), vec![&t_black, &t_white]);
  }
  save_tree(
    &t_black,
    std::fs::File::create("black.dot").unwrap(),
    100,
    5,
  );
  save_tree(&t_white, std::fs::File::create("white.dot").unwrap(), 100, 5);
}
