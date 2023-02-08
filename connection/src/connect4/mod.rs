use std::fmt::Display;

use fixedbitset::FixedBitSet;
use lib::FullyObservableDeterministicMctsProblem;

use crate::util::RectBitSet;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Color {
  Red = 0,
  Blue = 1,
}

#[derive(Clone)]
pub struct State<const H: usize, const W: usize> {
  board: [RectBitSet<H, W>; 2],
  heights: [u8; W],
  player_to_move: Color,
  remaining_tiles: u32,
  winner: Option<Color>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Move(pub u8);

pub struct C4<const H: usize, const W: usize>;

impl<const H: usize, const W: usize> FullyObservableDeterministicMctsProblem for C4<H, W> {
  type Agent = Color;
  type State = State<H, W>;
  type Action = Move;

  fn agent_to_act(&self, state: &Self::State) -> Self::Agent {
    state.player_to_move
  }

  fn agents(&self) -> Vec<Self::Agent> {
    vec![Color::Red, Color::Blue]
  }

  fn check_terminal(&self, state: &Self::State) -> bool {
    state.remaining_tiles == 0 || state.winner.is_some()
  }

  fn apply_action(&self, state: &mut Self::State, action: &Self::Action) -> Vec<f32> {
    let current_player_ix = state.player_to_move as usize;
    
    let col = action.0 as usize;
    let row = state.heights[col] as usize;
    //println!("dropped ({row},{col}) {:?}", state.player_to_move);

    state.board[current_player_ix].set((row, col), true);
    state.remaining_tiles -= 1;
    state.heights[col] += 1;
    
    let center = (row as i32, col as i32);
    let left = state.board[current_player_ix].ray_count(center, (0, -1), 3);
    let right = state.board[current_player_ix].ray_count(center, (0, 1), 3);
    let down = state.board[current_player_ix].ray_count(center, (-1, 0), 3);
    let up_left = state.board[current_player_ix].ray_count(center, (1, -1), 3);
    let up_right = state.board[current_player_ix].ray_count(center, (1, 1), 3);
    let down_left = state.board[current_player_ix].ray_count(center, (-1, -1), 3);
    let down_right = state.board[current_player_ix].ray_count(center, (-1, 1), 3);
  
    if down >= 3 || left + right >= 3 || up_left + down_right >= 3 || up_right + down_left >= 3 {
      //println!("end {left} {right} {down}");
      state.winner = Some(state.player_to_move);
      state.player_to_move.win_score()
    } else {
      state.player_to_move = state.player_to_move.opponent();
      vec![0.0, 0.0]
    }
  }

  fn start_state(&self) -> Self::State {
    State {
      player_to_move: Color::Red,
      winner: None,
      heights: [0; W],
      remaining_tiles: (H * W) as u32,
      board: [
        RectBitSet::new(),
        RectBitSet::new(),
      ],
    }
  }

  fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
    (0..W).filter(|col| state.heights[*col] < H as u8).map(|col| Move(col as u8)).collect()
  }
}

impl Into<u8> for Color {
  fn into(self) -> u8 {
    self as u8
  }
}

impl Color {
  fn opponent(&self) -> Color {
    match self {
        Color::Blue => Color::Red,
        Color::Red => Color::Blue
    }
  }

  fn win_score(&self) -> Vec<f32> {
    match self {
        Color::Red => vec![1.0, 0.0],
        Color::Blue => vec![0.0, 1.0]
    }
  }
}

impl Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<const H: usize, const W: usize> Display for State<H, W> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      for row in 0..H {
        for col in 0..W {
          if self.board[0][(H - 1 -row, col)] {
            write!(f, "R|")?;
          } else if self.board[1][(H - 1 - row, col)] {
            write!(f, "B|")?;
          } else {
            write!(f, " |")?;
          }
        }
        writeln!(f)?;
      }
      for col in 0..W {
        write!(f, "{}|", col % 10)?;
      }
      Ok(())
  }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::{rollout::RandomRollout, Expansion};

  #[test]
  fn test1() {
    let c4: C4<6, 7> = C4{};
    let t = RandomRollout(100);
    let states: Vec<_> = (0..10).map(|_| c4.start_state()).collect();
    let (values, policies) = t.block_expand(&c4, &states);
    println!("{:?}", values);
  }
}