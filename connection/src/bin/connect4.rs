use std::sync::Arc;
use text_io::read;

use connection::connect4::{C4, Move};
use lib::FullyObservableDeterministicMctsProblem;
use mcts::SearchLimit;
use mcts::search::Search;
use mcts::bandits::Uct;
use mcts::rollout::RandomRollout;

fn main() {
  let prompt = ">";
  let game: Arc<C4<6, 7> >= Arc::new(C4);
  let mut current_state = game.start_state();
  loop {
    print!("{}", prompt);
    let command: String = read!();
    match command.as_str() {
      "print" => {
        println!("{}", current_state);
      }
      "clear" => {
        current_state = game.start_state();
      }
      "drop" => {
        let col: u8 = read!();
        game.apply_action(&mut current_state, &Move(col));
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
          Uct(2.4),
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