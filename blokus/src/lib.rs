use std::collections::BTreeSet;

use lib::{
  mmdp::State as State_,
  utils::{Cell, Cell2D, Cells, Table},
};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Serialize, FromPrimitive)]
pub enum Player {
  Red = 0,
  Green = 1,
  Blue = 2,
  Yellow = 3,
}

pub struct TileInfo {
  id: u8,
  // true if the tile is mirror symm
  mirror_symm: bool,
  // rotation count
  rotation_count: u8,

  cells: Cells<2>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub enum Action {
  Pass,
  Place {
    offset: [i32; 2],
    flip: bool,
    tile_id: u8,
    rotate_count: u8,
  },
}

#[serde_as]
// player count
#[derive(Clone, Serialize)]
pub struct State<const H: usize, const W: usize, const P: usize> {
  table: Table<H, W, Option<Player>>,
  // todo: corners and edges
  player_to_move: Player,
  // for each player, store the remaining tiles
  #[serde_as(as = "[_; P]")]
  remaining_tiles: [BTreeSet<u8>; P],
  // corner cells
  #[serde_as(as = "[_; P]")]
  corner_cells: [Table<H, W, bool>; P],
  // adjacent cells
  #[serde_as(as = "[_; P]")]
  adjacent_cells: [Table<H, W, bool>; P],
}

enum StartCells {
  Corner,
  Center,
  Five,
}
pub struct Blokus<const H: usize, const W: usize, const P: usize> {
  start: StartCells,
}

impl<const H: usize, const W: usize, const P: usize> State_ for State<H, W, P> {
  type Action = Action;
  type Agent = Player;
  fn apply_action(&mut self, action: &Self::Action) -> Vec<f32> {
    match action {
      Action::Pass => {
        self.remaining_tiles[self.player_to_move as usize].clear();
        vec![0.0; P]
      }
      Action::Place {
        offset,
        flip,
        tile_id,
        rotate_count,
      } => {
        let tile_info = TileInfo::tile_info(*tile_id);
        for cell in tile_info.cells.iter() {
          self.table[cell] = Some(self.player_to_move);
        }
        let (corners, edges) = tile_info.cells.get_corners_and_edges();
        for cell in edges.iter() {
          let trans = cell + &Cell2D::from(offset);
          if trans[0] < 0 || trans[0] >= W as i32 || trans[1] < 0 || trans[1] >= H as i32 {
            continue;
          }
          if self.table[&trans].is_none() {
            self.adjacent_cells[self.player_to_move as usize][&trans] = true;
          }
        }
        for cell in corners.iter() {
          let trans = cell + &Cell2D::from(offset);
          if trans[0] < 0 || trans[0] >= W as i32 || trans[1] < 0 || trans[1] >= H as i32 {
            continue;
          }
          if !self.adjacent_cells[self.player_to_move as usize][trans] {
            self.corner_cells[self.player_to_move as usize][&trans] = true;
          }
        }
        vec![0.0; P]
      }
    }
  }

  fn current_agent(&self) -> Option<Self::Agent> {
    Some(self.player_to_move)
  }

  fn is_terminal(&self) -> bool {
    for p in 0..P {
      if !self.remaining_tiles[p].is_empty() {
        return false;
      }
    }
    return true;
  }

  fn legal_actions(&self) -> Vec<Self::Action> {
    let mut result = self.generate_valid_tile_placements(self.player_to_move);
    result.push(Action::Pass);
    result
  }
}

impl<const H: usize, const W: usize, const P: usize> State<H, W, P> {
  fn generate_valid_tile_placements(&self, player: Player) -> Vec<Action> {
    let mut result = vec![];
    let corners = &self.corner_cells[player as usize];
    let edge_adjacents = &self.adjacent_cells[player as usize];
    for tile_info in self.remaining_tiles[player as usize]
      .iter()
      .map(|id| TileInfo::tile_info(*id))
    {
      let cells = tile_info.cells;

      'rot: for rot_count in 0..4 {
        if rot_count == tile_info.rotation_count {
          break 'rot;
        }
        // rotate
        let mut after_rotation = cells.clone();
        for count in 0..rot_count {
          after_rotation = after_rotation.rotate();
        }

        'inner: for flip in [false, true] {
          if flip && tile_info.mirror_symm {
            // skip the second iteration for symmetric tiles
            break 'inner;
          }

          let cells = if flip {
            after_rotation.flip()
          } else {
            after_rotation.clone()
          };

          self.generate_valid_placements_for_tile(
            &mut result,
            player,
            tile_info.id,
            flip,
            rot_count,
            cells,
            corners,
            edge_adjacents,
          );
        }
      }
    }
    result
  }

  pub fn generate_valid_placements_for_tile(
    &self,
    result: &mut Vec<Action>,
    player: Player,
    tile_id: u8,
    flip: bool,
    rotate_count: u8,
    tile: Cells<2>,
    corners: &Table<H, W, bool>,
    edge_adjacents: &Table<H, W, bool>,
  ) {
    let bounds = tile.bounds();
    let x_span = bounds[0].1 - bounds[0].0;
    let y_span = bounds[1].1 - bounds[1].0;
    for x_offset in 0..(H as i32 - x_span + 1) {
      for y_offset in 0..(W as i32 - y_span + 1) {
        let offset = [x_offset - bounds[0].0, y_offset - bounds[1].0].into();
        if self.valid_placement(&tile, corners, edge_adjacents, &offset) {
          result.push(Action::Place {
            tile_id,
            offset: [x_offset - bounds[0].0, y_offset - bounds[1].0],
            flip,
            rotate_count,
          });
        }
      }
    }
  }

  fn total_scores(&self) -> [f32; P] {
    let mut result = [0.0; P];
    for y in 0..H as i32 {
      for x in 0..W as i32 {
        if self.table[&[x, y]].is_some() {
          let p = self.table[&[x, y]].unwrap();
          result[p as usize] += 1.0;
        }
      }
    }
    result
  }

  fn valid_placement(
    &self,
    tile: &Cells<2>,
    corners: &Table<H, W, bool>,
    edge_adjacents: &Table<H, W, bool>,
    offset: &Cell<2>,
  ) -> bool {
    let mut any_on_corner = false;
    // all cells should be empty
    // and at least one on corner
    for cell in tile.iter() {
      if self.table[cell + offset].is_some() {
        // overlapping
        return false;
      }

      if edge_adjacents[cell + offset] {
        return false;
      }

      if corners[cell + offset] {
        any_on_corner = true;
      }
    }

    if !any_on_corner {
      return false;
    }

    return true;
  }
}

impl<const H: usize, const W: usize, const P: usize> Blokus<H, W, P> {
  pub const fn new_corners() -> Self {
    Blokus {
      start: StartCells::Corner,
    }
  }

  pub const fn new_five() -> Self {
    Blokus {
      start: StartCells::Five,
    }
  }
}

impl TileInfo {
  fn new(id: u8, symm: bool, rc: u8, c: &[[i32; 2]]) -> Self {
    TileInfo {
      id,
      cells: c.into(),
      mirror_symm: symm,
      rotation_count: rc,
    }
  }

  fn tile_count() -> u8 {
    21
  }

  fn tile_info(ix: u8) -> TileInfo {
    match ix {
      // 1
      0 => TileInfo::new(0, true, 1, &[[0, 0]]),
      1 => TileInfo::new(1, true, 2, &[[0, 0], [0, 1]]),
      2 => TileInfo::new(2, true, 4, &[[0, 0], [0, 1], [1, 1]]),
      3 => TileInfo::new(3, true, 2, &[[0, 0], [0, 1], [0, 2]]),
      4 => TileInfo::new(4, true, 1, &[[0, 0], [0, 1], [1, 0], [1, 1]]),
      5 => TileInfo::new(5, true, 4, &[[0, 1], [1, 0], [1, 1], [1, 2]]),
      6 => TileInfo::new(6, true, 2, &[[0, 0], [0, 1], [0, 2], [0, 3]]),
      7 => TileInfo::new(7, false, 4, &[[0, 0], [0, 1], [0, 2], [-1, 2]]),
      8 => TileInfo::new(8, false, 2, &[[0, 0], [0, 1], [-1, 1], [-1, 2]]),
      9 => TileInfo::new(9, false, 4, &[[0, 0], [1, 0], [1, 1], [1, 2], [1, 3]]),
      10 => TileInfo::new(10, true, 4, &[[0, 0], [-2, 1], [-1, 1], [0, 1], [0, 2]]),
      11 => TileInfo::new(11, true, 4, &[[-2, 0], [-1, 0], [0, 0], [0, 1], [0, 2]]),
      12 => TileInfo::new(12, false, 4, &[[0, 0], [0, 1], [-1, 1], [-1, 2], [-1, 3]]),
      13 => TileInfo::new(13, false, 2, &[[0, 0], [-1, 0], [-1, 1], [-1, 2], [-2, 2]]),
      14 => TileInfo::new(14, true, 2, &[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),
      15 => TileInfo::new(15, false, 4, &[[0, 0], [1, 0], [2, 0], [1, 1], [2, 1]]),
      16 => TileInfo::new(16, true, 4, &[[1, 0], [2, 0], [0, 1], [1, 1], [0, 2]]),
      17 => TileInfo::new(17, true, 4, &[[0, 0], [1, 0], [2, 0], [0, 1], [2, 1]]),
      18 => TileInfo::new(18, false, 4, &[[1, 0], [0, 1], [1, 1], [2, 1], [0, 2]]),
      19 => TileInfo::new(19, true, 1, &[[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]]),
      20 => TileInfo::new(20, false, 4, &[[2, 0], [1, 1], [2, 1], [2, 2], [2, 3]]),
      _ => panic!("Unknow tile"),
    }
  }
}

impl TryFrom<usize> for Player {
  type Error = ();
  fn try_from(value: usize) -> Result<Self, Self::Error> {
    <Player as FromPrimitive>::from_usize(value).ok_or(())
  }
}

impl Into<usize> for Player {
  fn into(self) -> usize {
    self as usize
  }
}
