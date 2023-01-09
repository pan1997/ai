use std::ops::Index;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell<const D: usize> {
  coordinates: [i32; D],
}

impl Cell<2> {
  pub fn x(&self) -> i32 {
    self.coordinates[0]
  }

  pub fn y(&self) -> i32 {
    self.coordinates[1]
  }
}

impl<const D: usize> From<[i32; D]> for Cell<D> {
  fn from(coordinates: [i32; D]) -> Self {
    Cell { coordinates }
  }
}

pub struct Table<const H: usize, const W: usize, T> {
  cells: [[T; H]; W],
  x_min: i32,
  y_min: i32,
}

impl<const H: usize, const W: usize, T> Table<H, W, T>
where
  T: Copy,
{
  pub fn new(c: T, x_min: i32, y_min: i32) -> Self {
    Self {
      cells: [[c; H]; W],
      x_min,
      y_min,
    }
  }
}

impl<const H: usize, const W: usize, T> Index<Cell2D> for Table<H, W, T> {
  type Output = T;
  fn index(&self, index: Cell2D) -> &Self::Output {
    &self.cells[(index.x() - self.x_min) as usize][(index.y() - self.y_min) as usize]
  }
}

pub type Cell2D = Cell<2>;

struct BitBoard<T: num::Unsigned>(T);

impl<T: num::Unsigned> BitBoard<T> {
  fn new() -> Self {
    BitBoard(T::zero())
  }
}
