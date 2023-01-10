use std::{
  collections::BTreeSet,
  ops::{Add, Deref, DerefMut, Index, IndexMut},
};

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

#[serde_as]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Cell<const D: usize> {
  #[serde_as(as = "[_; D]")]
  coordinates: [i32; D],
}

#[serde_as]
#[derive(Serialize, Clone)]
pub struct Table<const H: usize, const W: usize, T: Serialize> {
  #[serde_as(as = "[[_; H]; W]")]
  cells: [[T; H]; W],
}

impl<const H: usize, const W: usize, T: Serialize> Table<H, W, T>
where
  T: Copy,
{
  pub fn new(c: T) -> Self {
    Self { cells: [[c; H]; W] }
  }
}

impl<const H: usize, const W: usize, T: Serialize> Index<Cell2D> for Table<H, W, T> {
  type Output = T;
  fn index(&self, index: Cell2D) -> &Self::Output {
    &self[&index]
  }
}

impl<const H: usize, const W: usize, T: Serialize> Index<&[i32; 2]> for Table<H, W, T> {
  type Output = T;
  fn index(&self, index: &[i32; 2]) -> &Self::Output {
    &self.cells[index[0] as usize][index[1] as usize]
  }
}

impl<const H: usize, const W: usize, T: Serialize> Index<&Cell2D> for Table<H, W, T> {
  type Output = T;
  fn index(&self, index: &Cell2D) -> &Self::Output {
    &self.cells[index.x() as usize][index.y() as usize]
  }
}

impl<const H: usize, const W: usize, T: Serialize> IndexMut<&Cell2D> for Table<H, W, T> {
  fn index_mut(&mut self, index: &Cell2D) -> &mut Self::Output {
    &mut self.cells[index.x() as usize][index.y() as usize]
  }
}

pub type Cell2D = Cell<2>;
impl Cell2D {
  pub fn x(&self) -> i32 {
    self.coordinates[0]
  }

  pub fn y(&self) -> i32 {
    self.coordinates[1]
  }

  pub fn adjacent_neighbours(&self) -> [Cell2D; 4] {
    [
      [self.coordinates[0] + 1, self.coordinates[1]].into(),
      [self.coordinates[0] - 1, self.coordinates[1]].into(),
      [self.coordinates[0], self.coordinates[1] + 1].into(),
      [self.coordinates[0], self.coordinates[1] - 1].into(),
    ]
  }

  pub fn diagonal_neighbours(&self) -> [Cell2D; 4] {
    [
      [self.coordinates[0] + 1, self.coordinates[1] + 1].into(),
      [self.coordinates[0] - 1, self.coordinates[1] - 1].into(),
      [self.coordinates[0] - 1, self.coordinates[1] + 1].into(),
      [self.coordinates[0] + 1, self.coordinates[1] - 1].into(),
    ]
  }
}

#[derive(Clone)]
pub struct Cells<const D: usize> {
  cells: Vec<Cell<D>>,
}

impl Cells<2> {
  pub fn flip(&self) -> Cells<2> {
    let mut cells = Vec::with_capacity(self.cells.len());
    for cell in self.cells.iter() {
      cells.push([cell.coordinates[0], -cell.coordinates[1]].into());
    }
    Self { cells }
  }

  pub fn rotate(&self) -> Cells<2> {
    let mut cells = Vec::with_capacity(self.cells.len());
    for cell in self.cells.iter() {
      cells.push([cell.coordinates[1], -cell.coordinates[0]].into());
    }
    Self { cells }
  }

  pub fn get_corners_and_edges(&self) -> (BTreeSet<Cell2D>, BTreeSet<Cell2D>) {
    let mut corners = BTreeSet::new();
    let mut edges = BTreeSet::new();
    for cell in self.iter() {
      for c in cell.adjacent_neighbours() {
        if !self.cells.contains(&c) {
          if corners.contains(&c) {
            corners.remove(&c);
          }
          edges.insert(c);
        }
      }
      for c in cell.diagonal_neighbours() {
        if !self.cells.contains(&c) {
          corners.insert(c);
        }
      }
    }
    (corners, edges)
  }
}

impl<const D: usize> Cells<D> {
  pub fn bounds(&self) -> Vec<(i32, i32)> {
    let mut result = vec![(i32::MAX, i32::MIN); D];
    for cell in self.cells.iter() {
      for d in 0..D {
        result[d].0 = std::cmp::min(result[d].0, cell.coordinates[d]);
        result[d].1 = std::cmp::max(result[d].1, cell.coordinates[d]);
      }
    }
    result
  }
}

impl<const D: usize, T: Into<Cell<D>> + Copy> From<&[T]> for Cells<D> {
  fn from(c: &[T]) -> Self {
    Cells {
      cells: c.iter().map(|x| <T as Into<Cell<D>>>::into(*x)).collect(),
    }
  }
}

impl<const D: usize> From<[i32; D]> for Cell<D> {
  fn from(coordinates: [i32; D]) -> Self {
    Cell { coordinates }
  }
}

impl<const D: usize> From<&[i32; D]> for Cell<D> {
  fn from(c: &[i32; D]) -> Self {
    Cell { coordinates: *c }
  }
}

impl<const D: usize> Deref for Cells<D> {
  type Target = [Cell<D>];
  fn deref(&self) -> &Self::Target {
    &self.cells
  }
}
impl<const D: usize> DerefMut for Cells<D> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.cells
  }
}

impl<const D: usize> Deref for Cell<D> {
  type Target = [i32; D];
  fn deref(&self) -> &Self::Target {
    &self.coordinates
  }
}

impl<const D: usize> DerefMut for Cell<D> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.coordinates
  }
}

impl<const H: usize, const W: usize, T: Serialize> Deref for Table<H, W, T> {
  type Target = [[T; H]; W];
  fn deref(&self) -> &Self::Target {
    &self.cells
  }
}

impl<const D: usize> Add for Cell<D> {
  type Output = Cell<D>;
  fn add(self, rhs: Self) -> Self::Output {
    let mut result = self;
    for ix in 0..D {
      result[ix] += rhs[ix];
    }
    result
  }
}

impl<const D: usize> Add for &Cell<D> {
  type Output = Cell<D>;
  fn add(self, rhs: Self) -> Self::Output {
    let mut result = *self;
    for ix in 0..D {
      result[ix] += rhs[ix];
    }
    result
  }
}
