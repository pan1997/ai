//! Core 2048 board representation, tile sliding mechanics, and chance tile spawn definitions.

/// Cardinal shift direction for sliding tiles on the 2048 grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Shift all tiles towards the left.
    Left,
    /// Shift all tiles towards the right.
    Right,
    /// Shift all tiles upwards.
    Up,
    /// Shift all tiles downwards.
    Down,
}

impl Direction {
    /// All four possible sliding directions.
    pub const ALL: [Direction; 4] = [
        Direction::Left,
        Direction::Right,
        Direction::Up,
        Direction::Down,
    ];

    /// Returns a short human-readable name (e.g. "Left", "Up").
    pub fn name(&self) -> &'static str {
        match self {
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Up => "Up",
            Self::Down => "Down",
        }
    }
}

/// Stochastic chance event representing a newly spawned tile on the board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileSpawn {
    /// Grid row index (0..4).
    pub row: u8,
    /// Grid column index (0..4).
    pub col: u8,
    /// Spawned tile value (typically 2 with 90% probability, or 4 with 10% probability).
    pub value: u16,
}

impl TileSpawn {
    /// Creates a new tile spawn description.
    pub const fn new(row: u8, col: u8, value: u16) -> Self {
        Self { row, col, value }
    }
}

/// State of a 2048 ($4 \times 4$) puzzle board.
///
/// Contains the grid tile values, cumulative score earned from tile merges,
/// and terminal status. Notice that this struct does not pin random number generator
/// state to allow stochastic chance-node branching during MCTS planning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tzf8State {
    /// $4 \times 4$ tile matrix (values are powers of 2: 2, 4, 8, ..., or 0 for empty).
    pub board: [[u32; 4]; 4],
    /// Cumulative score achieved through tile merges.
    pub score: u32,
    /// Flag indicating whether the game is currently ongoing.
    pub ongoing: bool,
}

impl Default for Tzf8State {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl Tzf8State {
    /// Creates a blank $4 \times 4$ board with 0 score.
    pub fn new_empty() -> Self {
        Self {
            board: [[0; 4]; 4],
            score: 0,
            ongoing: true,
        }
    }

    /// Creates a board initialized with two random tiles using `seed`.
    pub fn new(seed: u64) -> Self {
        crate::world::Tzf8World::initial_with_seed(seed)
    }

    /// Returns all coordinates `(row, col)` that currently contain 0 (empty).
    pub fn empty_cells(&self) -> Vec<(usize, usize)> {
        let mut buf = [(0u8, 0u8); 16];
        let count = self.empty_cells_buf(&mut buf);
        let mut cells = Vec::with_capacity(count);
        for &(r, c) in &buf[..count] {
            cells.push((r as usize, c as usize));
        }
        cells
    }

    /// Fills `out` with all coordinates `(row, col)` that currently contain 0 (empty) without heap allocation.
    ///
    /// Returns the number of empty cells written into `out`.
    #[inline]
    pub fn empty_cells_buf(&self, out: &mut [(u8, u8); 16]) -> usize {
        let mut count = 0;
        for r in 0..4 {
            for c in 0..4 {
                if self.board[r][c] == 0 {
                    out[count] = (r as u8, c as u8);
                    count += 1;
                }
            }
        }
        count
    }

    /// Returns the number of empty cells currently on the board without allocating memory.
    #[inline]
    pub fn count_empty_cells(&self) -> usize {
        let mut count = 0;
        for r in 0..4 {
            for c in 0..4 {
                if self.board[r][c] == 0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Returns the value of the highest tile on the board (e.g. 2048, 1024, 512).
    pub fn max_tile(&self) -> u32 {
        self.board.iter().flatten().copied().max().unwrap_or(0)
    }

    /// Applies a specific chance tile spawn to the board.
    pub fn apply_spawn(&mut self, spawn: TileSpawn) {
        let r = spawn.row as usize;
        let c = spawn.col as usize;
        assert!(
            r < 4 && c < 4,
            "TileSpawn coordinates out of bounds: ({r}, {c})"
        );
        assert_eq!(
            self.board[r][c], 0,
            "TileSpawn target cell ({r}, {c}) is not empty"
        );
        self.board[r][c] = spawn.value as u32;
    }

    /// Shifts a single 4-element row to the left, merging adjacent equal tiles.
    ///
    /// Returns `(changed, score_gain)`.
    fn shift_row_left(row: &mut [u32; 4]) -> (bool, u32) {
        let mut changed = false;
        let mut score_gain = 0;
        let mut compact = [0; 4];
        let mut pos = 0;

        for &val in row.iter() {
            if val != 0 {
                compact[pos] = val;
                pos += 1;
            }
        }

        let mut final_row = [0; 4];
        let mut fpos = 0;
        let mut i = 0;
        while i < 4 {
            if compact[i] == 0 {
                break;
            }
            if i + 1 < 4 && compact[i] == compact[i + 1] {
                let merged = compact[i] * 2;
                final_row[fpos] = merged;
                score_gain += merged;
                fpos += 1;
                i += 2;
            } else {
                final_row[fpos] = compact[i];
                fpos += 1;
                i += 1;
            }
        }

        if *row != final_row {
            changed = true;
            *row = final_row;
        }

        (changed, score_gain)
    }

    /// Transposes the $4 \times 4$ board matrix in-place.
    fn transpose(&mut self) {
        for r in 0..4 {
            for c in (r + 1)..4 {
                let t = self.board[r][c];
                self.board[r][c] = self.board[c][r];
                self.board[c][r] = t;
            }
        }
    }

    /// Shifts the board tiles in the specified direction.
    ///
    /// Returns `(changed, score_gain)`.
    pub fn move_board(&mut self, dir: Direction) -> (bool, u32) {
        let mut total_changed = false;
        let mut total_score = 0;

        match dir {
            Direction::Left => {
                for r in 0..4 {
                    let (changed, score) = Self::shift_row_left(&mut self.board[r]);
                    total_changed |= changed;
                    total_score += score;
                }
            }
            Direction::Right => {
                for r in 0..4 {
                    self.board[r].reverse();
                    let (changed, score) = Self::shift_row_left(&mut self.board[r]);
                    self.board[r].reverse();
                    total_changed |= changed;
                    total_score += score;
                }
            }
            Direction::Up => {
                self.transpose();
                for r in 0..4 {
                    let (changed, score) = Self::shift_row_left(&mut self.board[r]);
                    total_changed |= changed;
                    total_score += score;
                }
                self.transpose();
            }
            Direction::Down => {
                self.transpose();
                for r in 0..4 {
                    self.board[r].reverse();
                    let (changed, score) = Self::shift_row_left(&mut self.board[r]);
                    self.board[r].reverse();
                    total_changed |= changed;
                    total_score += score;
                }
                self.transpose();
            }
        }

        (total_changed, total_score)
    }

    /// Returns `true` if at least one legal move exists (empty cell or adjacent mergeable tiles).
    pub fn can_move(&self) -> bool {
        if !self.empty_cells().is_empty() {
            return true;
        }
        for r in 0..4 {
            for c in 0..4 {
                if (r + 1 < 4 && self.board[r][c] == self.board[r + 1][c])
                    || (c + 1 < 4 && self.board[r][c] == self.board[r][c + 1])
                {
                    return true;
                }
            }
        }
        false
    }

    /// Populates `out` with all directions that result in a board change.
    pub fn legal_directions(&self, out: &mut Vec<Direction>) {
        out.clear();
        for &dir in &Direction::ALL {
            let mut clone = self.clone();
            let (changed, _) = clone.move_board(dir);
            if changed {
                out.push(dir);
            }
        }
    }
}
