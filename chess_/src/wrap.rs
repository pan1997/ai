use std::fmt::Display;

use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};

#[derive(Clone)]
pub struct State {
  board: Board,
}

pub struct Game;

impl lib::mmdp::FODMMDP for Game {
  type Action = Action;
  type Agent = Player;
  type State = State;
  fn start_state(&self) -> Self::State {
    State {
      board: Board::default(),
    }
  }
  fn all_agents(&self) -> Vec<Self::Agent> {
    vec![Player(Color::White), Player(Color::Black)]
  }
}

impl lib::mmdp::State for State {
  type Agent = Player;
  type Action = Action;
  fn current_agent(&self) -> Option<Self::Agent> {
    Some(Player(self.board.side_to_move()))
  }

  fn is_terminal(&self) -> bool {
    self.board.status() != BoardStatus::Ongoing
  }

  fn legal_actions(&self) -> Vec<Self::Action> {
    MoveGen::new_legal(&self.board).map(|m| Action(m)).collect()
  }

  fn apply_action(&mut self, action: &Self::Action) -> Vec<f32> {
    let l = self.board.make_move_new(action.0);
    self.board = l;
    match self.board.status() {
      BoardStatus::Ongoing | BoardStatus::Stalemate => vec![0.0, 0.0],
      BoardStatus::Checkmate => match self.board.side_to_move() {
        Color::White => vec![0.0, 1.0],
        Color::Black => vec![1.0, 0.0],
      },
    }
  }
}

#[derive(Clone, Copy)]
pub struct Player(Color);

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Action(ChessMove);

impl TryFrom<usize> for Player {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    match value {
      0 => Ok(Player(Color::Black)),
      1 => Ok(Player(Color::White)),
      _ => Err(()),
    }
  }
}

impl Into<usize> for Player {
  fn into(self) -> usize {
    match self.0 {
      Color::Black => 0,
      Color::White => 1,
    }
  }
}

impl Display for Action {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}
