use mcts_traits::{AgentDynamics, Transition, World};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexPlayer {
    Black, // Connects Top to Bottom
    White, // Connects Left to Right
}

impl HexPlayer {
    pub fn other(self) -> Self {
        match self {
            HexPlayer::Black => HexPlayer::White,
            HexPlayer::White => HexPlayer::Black,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexState<const N: usize = 11> {
    pub board: Vec<Option<HexPlayer>>,
    pub current_player: HexPlayer,
}

impl<const N: usize> HexState<N> {
    pub fn new() -> Self {
        Self {
            board: vec![None; N * N],
            current_player: HexPlayer::Black,
        }
    }

    #[inline]
    pub fn idx(r: usize, c: usize) -> usize {
        r * N + c
    }

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

    pub fn check_win(&self, player: HexPlayer) -> bool {
        let mut visited = vec![false; N * N];
        let mut queue = VecDeque::new();

        match player {
            HexPlayer::Black => {
                // Top to Bottom: seed top row
                for c in 0..N {
                    let idx = Self::idx(0, c);
                    if self.board[idx] == Some(HexPlayer::Black) {
                        visited[idx] = true;
                        queue.push_back((0, c));
                    }
                }

                while let Some((r, c)) = queue.pop_front() {
                    if r == N - 1 {
                        return true;
                    }
                    for (nr, nc) in Self::neighbors(r, c) {
                        let n_idx = Self::idx(nr, nc);
                        if !visited[n_idx] && self.board[n_idx] == Some(HexPlayer::Black) {
                            visited[n_idx] = true;
                            queue.push_back((nr, nc));
                        }
                    }
                }
            }
            HexPlayer::White => {
                // Left to Right: seed left col
                for r in 0..N {
                    let idx = Self::idx(r, 0);
                    if self.board[idx] == Some(HexPlayer::White) {
                        visited[idx] = true;
                        queue.push_back((r, 0));
                    }
                }

                while let Some((r, c)) = queue.pop_front() {
                    if c == N - 1 {
                        return true;
                    }
                    for (nr, nc) in Self::neighbors(r, c) {
                        let n_idx = Self::idx(nr, nc);
                        if !visited[n_idx] && self.board[n_idx] == Some(HexPlayer::White) {
                            visited[n_idx] = true;
                            queue.push_back((nr, nc));
                        }
                    }
                }
            }
        }

        false
    }
}

impl<const N: usize> Default for HexState<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HexDynamics<const N: usize = 11>;

impl<const N: usize> AgentDynamics for HexDynamics<N> {
    type State = HexState<N>;
    type Action = usize;
    type Reward = [f32; 2];

    fn initial(&self) -> Self::State {
        HexState::new()
    }

    fn actions(&self, s: &Self::State) -> Vec<Self::Action> {
        (0..N * N).filter(|&idx| s.board[idx].is_none()).collect()
    }

    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward> {
        let idx = *action;
        assert!(idx < N * N && s.board[idx].is_none(), "Hex: invalid action");

        let mut next = s.clone();
        next.board[idx] = Some(s.current_player);

        if next.check_win(s.current_player) {
            let reward = match s.current_player {
                HexPlayer::Black => [1.0, -1.0],
                HexPlayer::White => [-1.0, 1.0],
            };
            Transition::new(next, reward, true)
        } else if next.board.iter().all(|c| c.is_some()) {
            Transition::new(next, [0.0, 0.0], true)
        } else {
            next.current_player = s.current_player.other();
            Transition::new(next, [0.0, 0.0], false)
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

    fn actions(&self, ws: &Self::WorldState, player: usize) -> Vec<Self::Action> {
        let active = match ws.current_player {
            HexPlayer::Black => 0,
            HexPlayer::White => 1,
        };
        if player == active {
            AgentDynamics::actions(self, ws)
        } else {
            Vec::new()
        }
    }

    fn step(
        &self,
        ws: Self::WorldState,
        joint: &[Self::Action],
    ) -> (Self::WorldState, Vec<f32>, bool) {
        let active = match ws.current_player {
            HexPlayer::Black => 0,
            HexPlayer::White => 1,
        };
        let t = AgentDynamics::step(self, ws, &joint[active]);
        (t.next_state, t.reward.to_vec(), t.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.check_win(HexPlayer::Black)
            || ws.check_win(HexPlayer::White)
            || ws.board.iter().all(|c| c.is_some())
    }
}

