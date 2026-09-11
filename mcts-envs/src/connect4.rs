use mcts_traits::{AgentDynamics, BatchedAgentDynamics, Transition, World};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Red,
    Yellow,
}

impl Player {
    #[inline]
    pub fn other(self) -> Self {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connect4State<const R: usize = 6, const C: usize = 7> {
    pub board: [[Option<Player>; C]; R],
    pub current_player: Player,
}

impl<const R: usize, const C: usize> Connect4State<R, C> {
    pub fn new() -> Self {
        Self {
            board: [[None; C]; R],
            current_player: Player::Red,
        }
    }

    pub fn is_column_full(&self, col: usize) -> bool {
        self.board[0][col].is_some()
    }

    pub fn is_board_full(&self) -> bool {
        (0..C).all(|c| self.is_column_full(c))
    }

    pub fn check_win_at(&self, r: usize, c: usize, p: Player) -> bool {
        // Horizontal
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

        // Vertical
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
}

impl<const R: usize, const C: usize> Default for Connect4State<R, C> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Connect4Dynamics<const R: usize = 6, const C: usize = 7>;

impl<const R: usize, const C: usize> AgentDynamics for Connect4Dynamics<R, C> {
    type State = Connect4State<R, C>;
    type Action = usize;
    type Reward = [f32; 2];

    fn initial(&self) -> Self::State {
        Connect4State::new()
    }

    fn actions(&self, s: &Self::State) -> Vec<Self::Action> {
        (0..C).filter(|&col| !s.is_column_full(col)).collect()
    }

    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward> {
        let col = *action;
        assert!(col < C, "Connect4: action column out of bounds");
        assert!(!s.is_column_full(col), "Connect4: column is full");

        let mut next_state = s.clone();
        let mut placed_row = 0;

        for row in (0..R).rev() {
            if next_state.board[row][col].is_none() {
                next_state.board[row][col] = Some(s.current_player);
                placed_row = row;
                break;
            }
        }

        let is_win = next_state.check_win_at(placed_row, col, s.current_player);
        if is_win {
            let reward = match s.current_player {
                Player::Red => [1.0, -1.0],
                Player::Yellow => [-1.0, 1.0],
            };
            Transition::new(next_state, reward, true)
        } else if next_state.is_board_full() {
            Transition::new(next_state, [0.0, 0.0], true)
        } else {
            next_state.current_player = s.current_player.other();
            Transition::new(next_state, [0.0, 0.0], false)
        }
    }
}

impl<const R: usize, const C: usize> BatchedAgentDynamics for Connect4Dynamics<R, C> {
    fn step_batch(
        &self,
        states: &[Self::State],
        actions: &[Self::Action],
        out_transitions: &mut Vec<Transition<Self::State, Self::Reward>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_transitions);
    }
}

impl<const R: usize, const C: usize> World for Connect4Dynamics<R, C> {
    type WorldState = Connect4State<R, C>;
    type Action = usize;
    type Observation = Connect4State<R, C>;

    fn n_players(&self) -> usize {
        2
    }

    fn initial(&self) -> Self::WorldState {
        Connect4State::new()
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        // Perfect information game: full state observation
        ws.clone()
    }

    fn actions(&self, ws: &Self::WorldState, player: usize) -> Vec<Self::Action> {
        let active_player = match ws.current_player {
            Player::Red => 0,
            Player::Yellow => 1,
        };
        if player == active_player {
            AgentDynamics::actions(self, ws)
        } else {
            Vec::new() // Inactive player has no moves
        }
    }

    fn step(
        &self,
        ws: Self::WorldState,
        joint: &[Self::Action],
    ) -> (Self::WorldState, Vec<f32>, bool) {
        let active_player = match ws.current_player {
            Player::Red => 0,
            Player::Yellow => 1,
        };
        let action = &joint[active_player];
        let transition = AgentDynamics::step(self, ws, action);
        (
            transition.next_state,
            transition.reward.to_vec(),
            transition.terminated,
        )
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.is_board_full()
            || (0..R).any(|r| (0..C).any(|c| ws.board[r][c].map(|p| ws.check_win_at(r, c, p)).unwrap_or(false)))
    }
}

