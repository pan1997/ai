//! Core TicTacToe game state representation, players, and rules.

/// Players in TicTacToe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    /// Player X: moves first (index 0).
    X,
    /// Player O: moves second (index 1).
    O,
}

impl Player {
    /// Returns the opponent player.
    #[inline]
    pub fn other(self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }

    /// Returns the 0-indexed player ID (X: 0, O: 1).
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Player::X => 0,
            Player::O => 1,
        }
    }

    /// Single-character symbol representation ('X' or 'O').
    #[inline]
    pub fn symbol(self) -> char {
        match self {
            Player::X => 'X',
            Player::O => 'O',
        }
    }
}

/// The 8 winning lines in a 3x3 grid (3 rows, 3 columns, 2 diagonals).
pub const WIN_LINES: [[usize; 3]; 8] = [
    // Rows
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    // Columns
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    // Diagonals
    [0, 4, 8],
    [2, 4, 6],
];

/// State representation of a 3x3 TicTacToe board.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TicTacToeState {
    /// Flat 9-cell board indexed row-major from 0 to 8.
    pub board: [Option<Player>; 9],
    /// Player whose turn it is to move.
    pub current_player: Player,
}

impl Default for TicTacToeState {
    fn default() -> Self {
        Self::new()
    }
}

impl TicTacToeState {
    /// Creates a new empty TicTacToe board with Player X to move.
    #[inline]
    pub fn new() -> Self {
        Self {
            board: [None; 9],
            current_player: Player::X,
        }
    }

    /// Returns `true` if the cell at index `idx` is empty.
    #[inline]
    pub fn is_empty(&self, idx: usize) -> bool {
        self.board[idx].is_none()
    }

    /// Returns `true` if all 9 cells are occupied.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.board.iter().all(|cell| cell.is_some())
    }

    /// Populates `out` with all legal empty cell indices (0..9).
    #[inline]
    pub fn legal_actions(&self, out: &mut Vec<usize>) {
        out.clear();
        if self.check_winner().is_some() {
            return;
        }
        for (i, cell) in self.board.iter().enumerate() {
            if cell.is_none() {
                out.push(i);
            }
        }
    }

    /// Populates `out` with the bitpacked action mask for legal actions.
    #[inline]
    pub fn action_mask(&self, out: &mut [u8]) {
        out.fill(0);
        let mut legal = Vec::new();
        self.legal_actions(&mut legal);
        for a in legal {
            if a / 8 < out.len() {
                out[a / 8] |= 1 << (a % 8);
            }
        }
    }

    /// Applies an action (occupying cell `action`), switching the current player.
    ///
    /// # Panics
    /// Panics if `action >= 9` or if the cell is already occupied.
    #[inline]
    pub fn apply_action(&mut self, action: usize) {
        assert!(action < 9, "Action {action} is out of bounds (0..9)");
        assert!(
            self.board[action].is_none(),
            "Cell {action} is already occupied"
        );
        self.board[action] = Some(self.current_player);
        self.current_player = self.current_player.other();
    }

    /// Checks if either player has achieved 3-in-a-row.
    #[inline]
    pub fn check_winner(&self) -> Option<Player> {
        for &[a, b, c] in &WIN_LINES {
            if let Some(p) = self.board[a] {
                if self.board[b] == Some(p) && self.board[c] == Some(p) {
                    return Some(p);
                }
            }
        }
        None
    }

    /// Returns `true` if the game has ended (either a win or full board draw).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.check_winner().is_some() || self.is_full()
    }

    /// Returns the terminal game outcome reward vector `[R_X, R_O]` if terminal.
    ///
    /// - Win for X: `Some([1.0, -1.0])`
    /// - Win for O: `Some([-1.0, 1.0])`
    /// - Draw: `Some([0.0, 0.0])`
    /// - Non-terminal: `None`
    #[inline]
    pub fn outcome(&self) -> Option<[f32; 2]> {
        if let Some(winner) = self.check_winner() {
            match winner {
                Player::X => Some([1.0, -1.0]),
                Player::O => Some([-1.0, 1.0]),
            }
        } else if self.is_full() {
            Some([0.0, 0.0])
        } else {
            None
        }
    }
}

impl mcts_traits::TensorRepresentable for TicTacToeState {
    const CHANNELS: usize = 2;
    const HEIGHT: usize = 3;
    const WIDTH: usize = 3;

    /// Encodes the 3x3 board into a perspective-normalized 2-channel float tensor:
    /// - Channel 0: Current player's pieces (1.0 if present, else 0.0).
    /// - Channel 1: Opponent's pieces (1.0 if present, else 0.0).
    #[inline]
    fn encode_tensor(&self, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            Self::CHANNELS * Self::HEIGHT * Self::WIDTH,
            "TicTacToeState::encode_tensor: output slice must be exactly 18 elements"
        );
        out.fill(0.0);

        let me = self.current_player;
        let opp = me.other();

        for i in 0..9 {
            if self.board[i] == Some(me) {
                out[i] = 1.0;
            } else if self.board[i] == Some(opp) {
                out[9 + i] = 1.0;
            }
        }
    }
}

