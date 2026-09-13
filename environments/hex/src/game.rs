//! Hex board state representation, Disjoint Set Union (DSU) path connectivity, and move rules.

/// Players in the game of Hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HexPlayer {
    /// Black player: connects the Top boundary to the Bottom boundary (moves first).
    Black,
    /// White player: connects the Left boundary to the Right boundary (moves second).
    White,
}

impl HexPlayer {
    /// Returns the opponent player.
    #[inline]
    pub const fn other(self) -> Self {
        match self {
            HexPlayer::Black => HexPlayer::White,
            HexPlayer::White => HexPlayer::Black,
        }
    }

    /// Returns 0 for Black, 1 for White.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            HexPlayer::Black => 0,
            HexPlayer::White => 1,
        }
    }

    /// Returns the display name of the player.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            HexPlayer::Black => "Black",
            HexPlayer::White => "White",
        }
    }

    /// Returns the single ASCII character representation for board rendering.
    #[inline]
    pub const fn char(self) -> char {
        match self {
            HexPlayer::Black => 'X',
            HexPlayer::White => 'O',
        }
    }

    /// Returns ANSI color escape code (Blue for Black, Red for White).
    #[inline]
    pub const fn color_code(self) -> &'static str {
        match self {
            HexPlayer::Black => "\x1b[1;34m", // Bold Blue
            HexPlayer::White => "\x1b[1;31m", // Bold Red
        }
    }
}

/// State of an $N \times N$ Hex board with Disjoint Set Union (DSU) win tracking.
///
/// Black connects Top (row 0) to Bottom (row $N-1$).
/// White connects Left (col 0) to Right (col $N-1$).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexState<const N: usize = 11> {
    /// Cell states stored in row-major order: `board[r * N + c]`.
    pub board: Vec<Option<HexPlayer>>,
    /// Current player to move.
    pub current_player: HexPlayer,
    /// Disjoint-set union tracking connectivity for Black.
    pub dsu_black: Vec<u16>,
    /// Disjoint-set union tracking connectivity for White.
    pub dsu_white: Vec<u16>,
    /// Whether the Pie (Swap) rule is active for this game.
    pub pie_rule: bool,
    /// Total number of half-moves (turns) played so far.
    pub move_count: usize,
}

impl<const N: usize> HexState<N> {
    /// Special action index representing the Pie (Swap) Rule invocation on Move 2.
    pub const SWAP_ACTION: usize = N * N;

    /// Constructs a new, empty $N \times N$ Hex board with Black to move.
    pub fn new() -> Self {
        assert!(
            N > 0 && N <= 128,
            "Hex board dimension N must be between 1 and 128"
        );
        let size = N * N + 2;
        let dsu_black: Vec<u16> = (0..size as u16).collect();
        let dsu_white: Vec<u16> = (0..size as u16).collect();
        Self {
            board: vec![None; N * N],
            current_player: HexPlayer::Black,
            dsu_black,
            dsu_white,
            pie_rule: false,
            move_count: 0,
        }
    }

    /// Constructs a new $N \times N$ Hex board with the Pie (Swap) rule optionally enabled.
    pub fn with_pie_rule(pie_rule: bool) -> Self {
        let mut s = Self::new();
        s.pie_rule = pie_rule;
        s
    }

    /// Returns the board dimension $N$.
    #[inline]
    pub const fn size(&self) -> usize {
        N
    }

    /// Converts 2D coordinates `(r, c)` to a flat 1D cell index.
    #[inline]
    pub const fn idx(r: usize, c: usize) -> usize {
        r * N + c
    }

    /// Converts a flat 1D cell index to 2D coordinates `(r, c)`.
    #[inline]
    pub const fn coord(idx: usize) -> (usize, usize) {
        (idx / N, idx % N)
    }

    /// Converts a flat 1D cell index to algebraic notation (e.g. `0 -> "A1"`, `(r=2, c=1) -> "B3"`).
    pub fn coord_to_str(idx: usize) -> String {
        if idx == Self::SWAP_ACTION {
            return "swap".to_string();
        }
        let (r, c) = Self::coord(idx);
        let col_char = (b'A' + c as u8) as char;
        format!("{}{}", col_char, r + 1)
    }

    /// Parses an algebraic coordinate (`"A1"`, `"f6"`), numeric pair (`"4 4"`), index (`"48"`), or `"swap"`.
    pub fn str_to_coord(s: &str) -> Option<usize> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return None;
        }

        if trimmed.eq_ignore_ascii_case("swap") {
            return Some(Self::SWAP_ACTION);
        }

        // Try parsing algebraic notation like "A1", "F6", "K11"
        let first = trimmed.chars().next()?;
        if first.is_ascii_alphabetic() {
            let col = (first.to_ascii_uppercase() as u8).checked_sub(b'A')? as usize;
            let row_str = &trimmed[first.len_utf8()..].trim();
            let row: usize = row_str.parse().ok()?;
            if row >= 1 && row <= N && col < N {
                return Some(Self::idx(row - 1, col));
            }
            return None;
        }

        // Try parsing row and col separated by whitespace or comma
        if let Some((r_part, c_part)) = trimmed.split_once(|c: char| c.is_whitespace() || c == ',')
        {
            let r: usize = r_part.trim().parse().ok()?;
            let c: usize = c_part.trim().parse().ok()?;
            if r < N && c < N {
                return Some(Self::idx(r, c));
            }
        }

        // Try parsing raw 1D index
        if let Ok(idx) = trimmed.parse::<usize>() {
            return (idx < N * N).then_some(idx);
        }

        None
    }

    /// Returns an iterator yielding valid neighboring cell coordinates on the hexagonal lattice.
    ///
    /// The 6 axial adjacency offsets are: `(-1, 0)`, `(-1, 1)`, `(0, -1)`, `(0, 1)`, `(1, -1)`, `(1, 0)`.
    #[inline]
    pub fn neighbors(r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
        const OFFSETS: [(isize, isize); 6] = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)];
        OFFSETS.into_iter().filter_map(move |(dr, dc)| {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nr < N as isize && nc >= 0 && nc < N as isize {
                Some((nr as usize, nc as usize))
            } else {
                None
            }
        })
    }

    /// Finds the canonical representative of `x` with path compression on a raw DSU slice.
    #[inline]
    pub fn dsu_find(dsu: &mut [u16], x: usize) -> usize {
        let mut root = x;
        while root != dsu[root] as usize {
            root = dsu[root] as usize;
        }
        let mut curr = x;
        while curr != root {
            let next = dsu[curr] as usize;
            dsu[curr] = root as u16;
            curr = next;
        }
        root
    }

    /// Merges sets containing `x` and `y` on a raw DSU slice.
    #[inline]
    pub fn dsu_union(dsu: &mut [u16], x: usize, y: usize) {
        let rx = Self::dsu_find(dsu, x);
        let ry = Self::dsu_find(dsu, y);
        if rx != ry {
            dsu[rx] = ry as u16;
        }
    }

    /// Finds the canonical representative of `x` with path compression.
    #[inline]
    pub fn find(&mut self, player: HexPlayer, x: usize) -> usize {
        match player {
            HexPlayer::Black => Self::dsu_find(&mut self.dsu_black, x),
            HexPlayer::White => Self::dsu_find(&mut self.dsu_white, x),
        }
    }

    /// Merges sets containing `x` and `y` in the player's DSU.
    #[inline]
    pub fn union(&mut self, player: HexPlayer, x: usize, y: usize) {
        match player {
            HexPlayer::Black => Self::dsu_union(&mut self.dsu_black, x, y),
            HexPlayer::White => Self::dsu_union(&mut self.dsu_white, x, y),
        }
    }

    /// Returns `true` if `player` has formed an unbroken connected path between their target boundaries.
    pub fn is_won(&self, player: HexPlayer) -> bool {
        let dsu = match player {
            HexPlayer::Black => &self.dsu_black,
            HexPlayer::White => &self.dsu_white,
        };
        let mut r1 = N * N;
        while r1 != dsu[r1] as usize {
            r1 = dsu[r1] as usize;
        }
        let mut r2 = N * N + 1;
        while r2 != dsu[r2] as usize {
            r2 = dsu[r2] as usize;
        }
        r1 == r2
    }

    /// Checks if `player` has won the game, compressing DSU paths in the process.
    pub fn check_win(&mut self, player: HexPlayer) -> bool {
        self.find(player, N * N) == self.find(player, N * N + 1)
    }

    /// Invokes the Pie (Swap) Rule on move 2.
    ///
    /// Finds the single Black stone on the board, clears it, and places a White stone
    /// at the transposed coordinates `(c, r)`. Resets Black's DSU and seeds White's DSU.
    pub fn play_swap(&mut self) {
        assert!(
            self.pie_rule && self.move_count == 1 && self.current_player == HexPlayer::White,
            "Hex: play_swap is only legal on Move 2 for White when the Pie Rule is enabled"
        );

        let black_idx = self
            .board
            .iter()
            .position(|&cell| cell == Some(HexPlayer::Black))
            .expect("Hex: exactly one Black stone must exist on Move 2");

        let (r, c) = Self::coord(black_idx);
        self.board[black_idx] = None;

        // Transpose across main diagonal: row becomes col, col becomes row
        let transposed_idx = Self::idx(c, r);
        self.board[transposed_idx] = Some(HexPlayer::White);

        // Reset DSUs
        let size = N * N + 2;
        self.dsu_black = (0..size as u16).collect();
        self.dsu_white = (0..size as u16).collect();

        // Connect transposed White stone to virtual Left/Right boundaries:
        // Left boundary for White is col 0 (which corresponds to r == 0 for Black)
        if r == 0 {
            self.union(HexPlayer::White, transposed_idx, N * N);
        }
        // Right boundary for White is col N - 1 (which corresponds to r == N - 1 for Black)
        if r == N - 1 {
            self.union(HexPlayer::White, transposed_idx, N * N + 1);
        }

        self.current_player = HexPlayer::Black;
        self.move_count += 1;
    }

    /// Places a stone at `idx` for the current player, or executes `SWAP_ACTION`.
    ///
    /// Returns `true` if this move immediately forms a winning connected path.
    pub fn play_move(&mut self, idx: usize) -> bool {
        if idx == Self::SWAP_ACTION {
            self.play_swap();
            return false;
        }

        self.move_count += 1;
        let player = self.current_player;
        self.board[idx] = Some(player);
        let (r, c) = Self::coord(idx);

        match player {
            HexPlayer::Black => {
                // Connect to virtual Top boundary (N * N)
                if r == 0 {
                    self.union(HexPlayer::Black, idx, N * N);
                }
                // Connect to virtual Bottom boundary (N * N + 1)
                if r == N - 1 {
                    self.union(HexPlayer::Black, idx, N * N + 1);
                }
                // Connect to neighboring Black stones
                for (nr, nc) in Self::neighbors(r, c) {
                    let n_idx = Self::idx(nr, nc);
                    if self.board[n_idx] == Some(HexPlayer::Black) {
                        self.union(HexPlayer::Black, idx, n_idx);
                    }
                }
                self.check_win(HexPlayer::Black)
            }
            HexPlayer::White => {
                // Connect to virtual Left boundary (N * N)
                if c == 0 {
                    self.union(HexPlayer::White, idx, N * N);
                }
                // Connect to virtual Right boundary (N * N + 1)
                if c == N - 1 {
                    self.union(HexPlayer::White, idx, N * N + 1);
                }
                // Connect to neighboring White stones
                for (nr, nc) in Self::neighbors(r, c) {
                    let n_idx = Self::idx(nr, nc);
                    if self.board[n_idx] == Some(HexPlayer::White) {
                        self.union(HexPlayer::White, idx, n_idx);
                    }
                }
                self.check_win(HexPlayer::White)
            }
        }
    }

    /// Populates `out` with all legal moves (unoccupied cell indices, plus `SWAP_ACTION` if eligible).
    pub fn legal_actions(&self, out: &mut Vec<usize>) {
        out.clear();
        for idx in 0..N * N {
            if self.board[idx].is_none() {
                out.push(idx);
            }
        }
        if self.pie_rule && self.move_count == 1 && self.current_player == HexPlayer::White {
            out.push(Self::SWAP_ACTION);
        }
    }

    /// Returns `true` if either player has won or all cells are filled.
    pub fn is_terminal(&self) -> bool {
        self.is_won(HexPlayer::Black)
            || self.is_won(HexPlayer::White)
            || self.board.iter().all(|c| c.is_some())
    }
}

impl<const N: usize> Default for HexState<N> {
    fn default() -> Self {
        Self::new()
    }
}
