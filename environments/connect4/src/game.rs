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

    /// Populates `out` with the bitpacked action mask for legal actions.
    #[inline]
    pub fn action_mask(&self, out: &mut [u8]) {
        out.fill(0);
        let mut legal = Vec::with_capacity(C);
        self.legal_actions(&mut legal);
        for a in legal {
            if a / 8 < out.len() {
                out[a / 8] |= 1 << (a % 8);
            }
        }
    }

    /// Drops a checker for `self.current_player` into `col` and alternates `self.current_player`.
    ///
    /// Returns the row index where the checker landed, and whether the move resulted in a win.
    #[inline]
    pub fn apply_action(&mut self, col: usize) -> (usize, bool) {
        let current = self.current_player;
        let row = self.drop_piece(col).expect("Connect4: illegal action");
        let is_win = self.check_win_at(row, col, current);
        self.current_player = current.other();
        (row, is_win)
    }

    /// Returns `true` if the state is terminal (board is full or either player has won).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.is_board_full() || self.has_won(Player::Red) || self.has_won(Player::Yellow)
    }

    /// Returns the winning player if one exists.
    #[inline]
    pub fn check_winner(&self) -> Option<Player> {
        if self.has_won(Player::Red) {
            Some(Player::Red)
        } else if self.has_won(Player::Yellow) {
            Some(Player::Yellow)
        } else {
            None
        }
    }
}

impl<const R: usize, const C: usize> Default for Connect4State<R, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const R: usize, const C: usize> mcts_traits::TensorRepresentable for Connect4State<R, C> {
    const CHANNELS: usize = 3;
    const HEIGHT: usize = R;
    const WIDTH: usize = C;

    /// Encodes the $R \times C$ board into an absolute fixed-seat 3-channel float tensor:
    /// - Channel 0: Player Red checkers (1.0 if present, else 0.0).
    /// - Channel 1: Player Yellow checkers (1.0 if present, else 0.0).
    /// - Channel 2: Turn indicator (1.0 if Red to move, else 0.0).
    #[inline]
    fn encode_tensor(&self, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            Self::CHANNELS * Self::HEIGHT * Self::WIDTH,
            "Connect4State::encode_tensor: output slice must be exactly {} elements",
            Self::CHANNELS * Self::HEIGHT * Self::WIDTH
        );
        out.fill(0.0);

        let plane_size = R * C;
        let turn_val = if self.current_player == Player::Red { 1.0 } else { 0.0 };

        for r in 0..R {
            for c in 0..C {
                let idx = r * C + c;
                if self.board[r][c] == Some(Player::Red) {
                    out[idx] = 1.0;
                } else if self.board[r][c] == Some(Player::Yellow) {
                    out[plane_size + idx] = 1.0;
                }
                out[2 * plane_size + idx] = turn_val;
            }
        }
    }
}

impl<const R: usize, const C: usize> mcts_onnx::SelfPlayEnv for Connect4State<R, C> {
    type Dynamics = mcts_traits::TurnBasedDynamics<crate::Connect4World<R, C>>;

    fn dynamics(&self) -> Self::Dynamics {
        mcts_traits::TurnBasedDynamics::new(crate::Connect4World::<R, C>::new())
    }

    fn initial() -> Self {
        Self::new()
    }

    fn legal_actions(&self, out: &mut Vec<usize>) {
        self.legal_actions(out);
    }

    fn action_mask(&self, out: &mut [u8]) {
        self.action_mask(out);
    }

    fn apply_action(&mut self, action: usize) {
        self.apply_action(action);
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal()
    }

    fn terminal_returns(&self) -> [f32; 2] {
        match self.check_winner() {
            Some(Player::Red) => [1.0, -1.0],
            Some(Player::Yellow) => [-1.0, 1.0],
            None => [0.0, 0.0],
        }
    }

    fn current_player_index(&self) -> usize {
        self.current_player.index()
    }
}



