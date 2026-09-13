//! Core Connect 4 game state representation, players, and rules.

/// Players in Connect 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    /// Red player: moves first (index 0).
    Red,
    /// Yellow player: moves second (index 1).
    Yellow,
}

impl Player {
    /// Returns the opponent player.
    #[inline]
    pub fn other(self) -> Self {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }

    /// Returns the 0-indexed player ID (Red: 0, Yellow: 1).
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Player::Red => 0,
            Player::Yellow => 1,
        }
    }

    /// Single-character symbol representation (`R` or `Y`).
    #[inline]
    pub fn symbol(self) -> char {
        match self {
            Player::Red => 'R',
            Player::Yellow => 'Y',
        }
    }
}

/// State representation of a Connect 4 board of size $R \times C$.
///
/// Default dimensions are standard tournament size: 6 rows by 7 columns ($R=6, C=7$).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Connect4State<const R: usize = 6, const C: usize = 7> {
    /// Grid cells where `[0][col]` is the top cell and `[R-1][col]` is the bottom cell.
    pub board: [[Option<Player>; C]; R],
    /// Player whose turn it is to drop a checker.
    pub current_player: Player,
}

impl<const R: usize, const C: usize> Connect4State<R, C> {
    /// Creates a new, empty Connect 4 board with `Player::Red` to move.
    pub fn new() -> Self {
        Self {
            board: [[None; C]; R],
            current_player: Player::Red,
        }
    }

    /// Returns `true` if the specified column cannot accept further checkers.
    #[inline]
    pub fn is_column_full(&self, col: usize) -> bool {
        self.board[0][col].is_some()
    }

    /// Returns `true` if all columns are filled to capacity.
    #[inline]
    pub fn is_board_full(&self) -> bool {
        (0..C).all(|c| self.is_column_full(c))
    }

    /// Populates `out` with all legal columns that are not yet full.
    #[inline]
    pub fn legal_actions(&self, out: &mut Vec<usize>) {
        out.clear();
        for col in 0..C {
            if !self.is_column_full(col) {
                out.push(col);
            }
        }
    }

    /// Drops a checker for `self.current_player` into `col`.
    ///
    /// Returns the row index `0..R` where the checker came to rest, or an error if the column is full or invalid.
    pub fn drop_piece(&mut self, col: usize) -> Result<usize, &'static str> {
        if col >= C {
            return Err("Column index out of bounds");
        }
        if self.is_column_full(col) {
            return Err("Column is already full");
        }

        for row in (0..R).rev() {
            if self.board[row][col].is_none() {
                self.board[row][col] = Some(self.current_player);
                return Ok(row);
            }
        }

        Err("No available row in column")
    }

    /// Checks if placing a checker for player `p` at `(r, c)` forms a winning line of 4.
    pub fn check_win_at(&self, r: usize, c: usize, p: Player) -> bool {
        // Horizontal (-)
        let mut count = 0;
        let start_c = c.saturating_sub(3);
        let end_c = std::cmp::min(c + 3, C - 1);
        for col in start_c..=end_c {
            if self.board[r][col] == Some(p) {
                count += 1;
                if count >= 4 {
                    return true;
                }
            } else {
                count = 0;
            }
        }

        // Vertical (|)
        count = 0;
        let start_r = r.saturating_sub(3);
        let end_r = std::cmp::min(r + 3, R - 1);
        for row in start_r..=end_r {
            if self.board[row][c] == Some(p) {
                count += 1;
                if count >= 4 {
                    return true;
                }
            } else {
                count = 0;
            }
        }

        // Diagonal down-right (\)
        count = 0;
        let mut steps = 0;
        while r > steps && c > steps && steps < 3 {
            steps += 1;
        }
        let mut row = r - steps;
        let mut col = c - steps;
        while row < R && col < C {
            if col > c + 3 {
                break;
            }
            if self.board[row][col] == Some(p) {
                count += 1;
                if count >= 4 {
                    return true;
                }
            } else {
                count = 0;
            }
            row += 1;
            col += 1;
        }

        // Diagonal up-right (/)
        count = 0;
        let mut steps = 0;
        while r + steps + 1 < R && c > steps && steps < 3 {
            steps += 1;
        }
        let mut row = r + steps;
        let mut col = c - steps;
        while col < C {
            if col > c + 3 {
                break;
            }
            if self.board[row][col] == Some(p) {
                count += 1;
                if count >= 4 {
                    return true;
                }
            } else {
                count = 0;
            }
            if row == 0 {
                break;
            }
            row -= 1;
            col += 1;
        }

        false
    }

    /// Checks if player `p` has at least one connected line of 4 anywhere on the board.
    pub fn has_won(&self, p: Player) -> bool {
        for r in 0..R {
            for c in 0..C {
                if self.board[r][c] == Some(p) && self.check_win_at(r, c, p) {
                    return true;
                }
            }
        }
        false
    }
}

impl<const R: usize, const C: usize> Default for Connect4State<R, C> {
    fn default() -> Self {
        Self::new()
    }
}
