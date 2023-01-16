use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};
use lib::FullyObservableDeterministicMctsProblem;

pub struct Game;

#[derive(Clone, Copy)]
pub struct Player(Color);

impl FullyObservableDeterministicMctsProblem for Game {
  type Agent = Player;
  type Action = ChessMove;
  type State = Board;

  fn agent_to_act(&self, state: &Self::State) -> Self::Agent {
    Player(state.side_to_move())
  }

  fn apply_action(&self, state: &mut Self::State, action: &Self::Action) -> Vec<f32> {
    let l = state.make_move_new(*action);
    *state = l;
    match state.status() {
      BoardStatus::Ongoing | BoardStatus::Stalemate => vec![0.0, 0.0],
      BoardStatus::Checkmate => match state.side_to_move() {
        Color::White => vec![0.0, 1.0],
        Color::Black => vec![1.0, 0.0],
      },
    }
  }

  fn start_state(&self) -> Self::State {
    Board::default()
  }

  fn check_terminal(&self, state: &Self::State) -> bool {
    state.status() != BoardStatus::Ongoing
  }

  fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
    MoveGen::new_legal(state).collect()
  }

  fn agents(&self) -> Vec<Self::Agent> {
    vec![Player(Color::White), Player(Color::Black)]
  }
}

impl TryFrom<u8> for Player {
  type Error = ();
  fn try_from(value: u8) -> Result<Self, Self::Error> {
    match value {
      0 => Ok(Player(Color::Black)),
      1 => Ok(Player(Color::White)),
      _ => Err(()),
    }
  }
}

impl Into<u8> for Player {
  fn into(self) -> u8 {
    match self.0 {
      Color::Black => 0,
      Color::White => 1,
    }
  }
}
