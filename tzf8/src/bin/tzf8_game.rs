use tzf8::{Tzf8, State, Move};
use lib::MctsProblem;
use text_io::read;
use std::sync::Arc;
use mcts::{SearchLimit, search::Search, rollout::RandomRollout, bandits::Uct};

fn main() {
  let game = Arc::new(Tzf8);
  let mut current_state = State::new();
  println!("{current_state}");
  let r: u8 = read!();
  let c: u8 = read!();
  let v: u32 = read!();
  current_state.place(v, 3 - r, c);
  println!("{current_state}");
  let r: u8 = read!();
  let c: u8 = read!();
  let v: u32 = read!();
  current_state.place(v, 3 - r, c);
  while !game.check_terminal(&current_state) {
    println!("{current_state}");
    let lim = 100000;
    let limit = SearchLimit::new(lim);
    let search = Search::new(
      game.clone(),
      Arc::new(current_state.clone()),
      1,
      limit,
      Uct(1.2),
      RandomRollout(50),
    );
    let mut worker = search.create_workers(1);
    search.start(&mut worker[0]);
    let policy = search.get_policy();
    for (a, s, v) in policy {
      println!("{a} -> prob {s:.5}, value: {v:.5}");
    }
    let m: String = read!();
    match m.as_str() {
      "l" => {
        current_state.apply_move(&Move::Left);
      }
      "r" => {
        current_state.apply_move(&Move::Right);
      }
      "u" => {
        current_state.apply_move(&Move::Up);
      }
      "d" => {
        current_state.apply_move(&Move::Down);
      },
      _ => {panic!("unknown move")}
    }
    let r: u8 = read!();
    let c: u8 = read!();
    let v: u32 = read!();
    current_state.place(v, 3 - r, c);
  }
}