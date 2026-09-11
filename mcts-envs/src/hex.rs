use mcts_traits::{AgentDynamics, StepOutcome, World};

/// Players in the game of Hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexPlayer {
    /// Black player: connects the Top boundary to the Bottom boundary (moves first).
    Black,
    /// White player: connects the Left boundary to the Right boundary (moves second).
    White,
}

impl HexPlayer {
    /// Returns the opponent player.
    pub fn other(self) -> Self {
        match self {
            HexPlayer::Black => HexPlayer::White,
            HexPlayer::White => HexPlayer::Black,
        }
    }
}

/// State of an $N \times N$ Hex board with Disjoint Set Union (DSU) win tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexState<const N: usize = 11> {
    /// Cell states stored in row-major order: `board[r * N + c]`.
    pub board: Vec<Option<HexPlayer>>,
    /// Current player to move.
    pub current_player: HexPlayer,
    dsu_black: Vec<u16>,
    dsu_white: Vec<u16>,
}

impl<const N: usize> HexState<N> {
    /// Constructs a new, empty $N \times N$ Hex board with Black to move.
    pub fn new() -> Self {
        let size = N * N + 2;
        let dsu_black: Vec<u16> = (0..size as u16).collect();
        let dsu_white: Vec<u16> = (0..size as u16).collect();
        Self {
            board: vec![None; N * N],
            current_player: HexPlayer::Black,
            dsu_black,
            dsu_white,
        }
    }

    /// Converts 2D coordinates `(r, c)` to a flat 1D cell index.
    #[inline]
    pub fn idx(r: usize, c: usize) -> usize {
        r * N + c
    }

    /// Returns an iterator yielding valid neighboring cell coordinates on the hexagonal lattice.
    #[inline]
    pub fn neighbors(r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
        let offsets = [
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
        ];
        offsets.into_iter().filter_map(move |(dr, dc)| {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nr < N as isize && nc >= 0 && nc < N as isize {
                Some((nr as usize, nc as usize))
            } else {
                None
            }
        })
    }

    fn find_internal(dsu: &mut [u16], x: usize) -> usize {
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

    /// Finds the canonical representative of `x` with path compression.
    pub fn find(&mut self, player: HexPlayer, x: usize) -> usize {
        match player {
            HexPlayer::Black => Self::find_internal(&mut self.dsu_black, x),
            HexPlayer::White => Self::find_internal(&mut self.dsu_white, x),
        }
    }

    /// Merges sets containing `x` and `y` in the player's DSU.
    pub fn union(&mut self, player: HexPlayer, x: usize, y: usize) {
        let rx = self.find(player, x);
        let ry = self.find(player, y);
        if rx != ry {
            match player {
                HexPlayer::Black => self.dsu_black[rx] = ry as u16,
                HexPlayer::White => self.dsu_white[rx] = ry as u16,
            }
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

    /// Places a stone at `idx` for the current player and updates DSU boundary connections.
    ///
    /// Returns `true` if this move immediately forms a winning connected path.
    pub fn play_move(&mut self, idx: usize) -> bool {
        let player = self.current_player;
        self.board[idx] = Some(player);
        let r = idx / N;
        let c = idx % N;

        match player {
            HexPlayer::Black => {
                if r == 0 {
                    self.union(HexPlayer::Black, idx, N * N);
                }
                if r == N - 1 {
                    self.union(HexPlayer::Black, idx, N * N + 1);
                }
                for (nr, nc) in Self::neighbors(r, c) {
                    let n_idx = Self::idx(nr, nc);
                    if self.board[n_idx] == Some(HexPlayer::Black) {
                        self.union(HexPlayer::Black, idx, n_idx);
                    }
                }
                self.check_win(HexPlayer::Black)
            }
            HexPlayer::White => {
                if c == 0 {
                    self.union(HexPlayer::White, idx, N * N);
                }
                if c == N - 1 {
                    self.union(HexPlayer::White, idx, N * N + 1);
                }
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
}

impl<const N: usize> Default for HexState<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Parametric Hex game dynamics on an $N \times N$ board.
///
/// Implements:
/// - [`AgentDynamics`]: Planning state transitions and legal cell generation.
/// - [`World`]: 2-player match referee with path-connectivity detection.
#[derive(Debug, Clone, Copy, Default)]
pub struct HexDynamics<const N: usize = 11>;

impl<const N: usize> AgentDynamics for HexDynamics<N> {
    type State = HexState<N>;
    type Action = usize;
    type Reward = [f32; 2];

    fn initial(&self) -> Self::State {
        HexState::new()
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        for idx in 0..N * N {
            if s.board[idx].is_none() {
                out.push(idx);
            }
        }
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        let idx = *action;
        assert!(idx < N * N && s.board[idx].is_none(), "Hex: invalid action");
        let player = s.current_player;
        let won = s.play_move(idx);

        if won {
            let reward = match player {
                HexPlayer::Black => [1.0, -1.0],
                HexPlayer::White => [-1.0, 1.0],
            };
            StepOutcome::new(reward, true)
        } else if s.board.iter().all(|c| c.is_some()) {
            StepOutcome::new([0.0, 0.0], true)
        } else {
            s.current_player = player.other();
            StepOutcome::new([0.0, 0.0], false)
        }
    }
}

impl<const N: usize> World for HexDynamics<N> {
    type WorldState = HexState<N>;
    type Action = usize;
    type Observation = HexState<N>;

    fn n_players(&self) -> usize {
        2
    }

    fn initial(&self) -> Self::WorldState {
        HexState::new()
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        ws.clone()
    }

    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        let active = match ws.current_player {
            HexPlayer::Black => 0,
            HexPlayer::White => 1,
        };
        if player == active {
            AgentDynamics::actions(self, ws, out);
        }
    }

    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool) {
        let active = match ws.current_player {
            HexPlayer::Black => 0,
            HexPlayer::White => 1,
        };
        let outcome = AgentDynamics::step(self, ws, &joint[active]);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.is_won(HexPlayer::Black)
            || ws.is_won(HexPlayer::White)
            || ws.board.iter().all(|c| c.is_some())
    }
}

