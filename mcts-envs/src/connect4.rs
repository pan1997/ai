use mcts_traits::{AgentDynamics, BatchedAgentDynamics, StepOutcome, World};

/// Players in Connect 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    /// First player (moves first, index 0).
    Red,
    /// Second player (moves second, index 1).
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
}

/// State representation of a Connect 4 board of size $R \times C$.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn is_column_full(&self, col: usize) -> bool {
        self.board[0][col].is_some()
    }

    /// Returns `true` if all columns are filled to capacity.
    pub fn is_board_full(&self) -> bool {
        (0..C).all(|c| self.is_column_full(c))
    }

    /// Checks if placing a checker for player `p` at `(r, c)` forms a line of 4.
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

/// Parametric Connect 4 game dynamics with $R$ rows and $C$ columns.
///
/// Implements:
/// - [`AgentDynamics`]: Single-player planning transition steps.
/// - [`BatchedAgentDynamics`]: Vectorized batch steps.
/// - [`World`]: 2-player match referee with simultaneous joint-action steps.
#[derive(Debug, Clone, Copy, Default)]
pub struct Connect4Dynamics<const R: usize = 6, const C: usize = 7>;

impl<const R: usize, const C: usize> AgentDynamics for Connect4Dynamics<R, C> {
    type State = Connect4State<R, C>;
    type Action = usize;
    type Reward = [f32; 2];

    fn initial(&self) -> Self::State {
        Connect4State::new()
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        for col in 0..C {
            if !s.is_column_full(col) {
                out.push(col);
            }
        }
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        let col = *action;
        assert!(col < C, "Connect4: action column out of bounds");
        assert!(!s.is_column_full(col), "Connect4: column is full");

        let mut placed_row = 0;
        let current_player = s.current_player;

        for row in (0..R).rev() {
            if s.board[row][col].is_none() {
                s.board[row][col] = Some(current_player);
                placed_row = row;
                break;
            }
        }

        let is_win = s.check_win_at(placed_row, col, current_player);
        if is_win {
            let reward = match current_player {
                Player::Red => [1.0, -1.0],
                Player::Yellow => [-1.0, 1.0],
            };
            StepOutcome::new(reward, true)
        } else if s.is_board_full() {
            StepOutcome::new([0.0, 0.0], true)
        } else {
            s.current_player = current_player.other();
            StepOutcome::new([0.0, 0.0], false)
        }
    }
}

impl<const R: usize, const C: usize> BatchedAgentDynamics for Connect4Dynamics<R, C> {
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, out_outcomes);
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

    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        let active_player = match ws.current_player {
            Player::Red => 0,
            Player::Yellow => 1,
        };
        if player == active_player {
            AgentDynamics::actions(self, ws, out);
        }
    }

    fn step(
        &self,
        ws: &mut Self::WorldState,
        joint: &[Self::Action],
    ) -> (Vec<f32>, bool) {
        let active_player = match ws.current_player {
            Player::Red => 0,
            Player::Yellow => 1,
        };
        let action = &joint[active_player];
        let outcome = AgentDynamics::step(self, ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.is_board_full()
            || (0..R).any(|r| (0..C).any(|c| ws.board[r][c].map(|p| ws.check_win_at(r, c, p)).unwrap_or(false)))
    }
}

