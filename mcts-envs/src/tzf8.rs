use mcts_traits::{AgentDynamics, Transition};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tzf8State {
    pub board: [[u32; 4]; 4],
    pub score: u32,
    pub ongoing: bool,
    rng_seed: u64,
}

impl Tzf8State {
    pub fn new(seed: u64) -> Self {
        let mut state = Self {
            board: [[0; 4]; 4],
            score: 0,
            ongoing: true,
            rng_seed: if seed == 0 { 0xda942042e4dd58b5 } else { seed },
        };
        state.add_random_tile();
        state.add_random_tile();
        state
    }

    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_seed = x;
        x
    }

    pub fn empty_cells(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::new();
        for r in 0..4 {
            for c in 0..4 {
                if self.board[r][c] == 0 {
                    cells.push((r, c));
                }
            }
        }
        cells
    }

    pub fn add_random_tile(&mut self) -> bool {
        let empty = self.empty_cells();
        if empty.is_empty() {
            return false;
        }
        let rand_val = self.next_rand();
        let idx = (rand_val as usize) % empty.len();
        let (r, c) = empty[idx];
        let val = if (rand_val >> 16).is_multiple_of(10) { 4 } else { 2 };
        self.board[r][c] = val;
        true
    }

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

    fn transpose(&mut self) {
        for r in 0..4 {
            for c in (r + 1)..4 {
                let t = self.board[r][c];
                self.board[r][c] = self.board[c][r];
                self.board[c][r] = t;
            }
        }
    }

    pub fn can_move(&self) -> bool {
        if !self.empty_cells().is_empty() {
            return true;
        }
        for r in 0..4 {
            for c in 0..4 {
                let val = self.board[r][c];
                if (r + 1 < 4 && self.board[r + 1][c] == val)
                    || (c + 1 < 4 && self.board[r][c + 1] == val)
                {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Tzf8Dynamics;

impl AgentDynamics for Tzf8Dynamics {
    type State = Tzf8State;
    type Action = Direction;
    type Reward = [f32; 1];

    fn initial(&self) -> Self::State {
        Tzf8State::new(12345)
    }

    fn actions(&self, s: &Self::State) -> Vec<Self::Action> {
        let directions = [
            Direction::Left,
            Direction::Right,
            Direction::Up,
            Direction::Down,
        ];
        directions
            .into_iter()
            .filter(|&dir| {
                let mut test_state = s.clone();
                let (changed, _) = test_state.move_board(dir);
                changed
            })
            .collect()
    }

    fn step(&self, mut s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward> {
        let (changed, score_gain) = s.move_board(*action);
        if !changed {
            s.ongoing = false;
            return Transition::new(s, [0.0], true);
        }

        s.score += score_gain;
        s.add_random_tile();

        let terminated = !s.can_move();
        if terminated {
            s.ongoing = false;
        }

        Transition::new(s, [score_gain as f32], terminated)
    }
}
