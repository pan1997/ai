use std::sync::Arc;

use mcts::{bandits::Uct, rollout::RandomRollout, search::Search, SearchLimit};
use text_io::read;
use tzf8::{Move, State, Tzf8};

fn main() {
  let prompt = ">";
  let game = Arc::new(Tzf8);
  let mut current_state = State::new();
  loop {
    print!("{}", prompt);
    let command: String = read!();
    match command.as_str() {
      "print" => {
        println!("{}", current_state);
      }
      "clear" => {
        current_state = State::new();
      }
      "drop" => {
        let r: u8 = read!();
        let c: u8 = read!();
        let v: u32 = read!();
        current_state.place(v, 3 - r, c);
      }
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
      }
      "analyse" => {
        //let lim = read!();
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
      }
      "exit" | "quit" | "bye" => {
        return;
      }
      _ => {}
    }
  }
}
