//! Ground-truth referee and environment driver for 2048 matches.

use crate::game::{Direction, TileSpawn, Tzf8State};
use mcts_traits::{StepOutcome, World};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

/// Impartial referee and match driver for 2048.
///
/// Manages physical tile moves and stochastic tile insertions using a pseudorandom number generator.
pub struct Tzf8World {
    rng: RefCell<StdRng>,
}

impl Default for Tzf8World {
    fn default() -> Self {
        Self::new()
    }
}

impl Tzf8World {
    /// Creates a new `Tzf8World` seeded from system entropy.
    pub fn new() -> Self {
        Self {
            rng: RefCell::new(StdRng::from_entropy()),
        }
    }

    /// Creates a new `Tzf8World` with a deterministic seed for reproducible benchmarking.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Samples a stochastic tile spawn on `state` using the referee's random generator.
    ///
    /// 90% probability for tile value 2, 10% probability for tile value 4.
    pub fn sample_spawn(&self, state: &Tzf8State) -> Option<TileSpawn> {
        let empty = state.empty_cells();
        if empty.is_empty() {
            return None;
        }
        let mut rng = self.rng.borrow_mut();
        let idx = rng.gen_range(0..empty.len());
        let (row, col) = empty[idx];
        let value = if rng.gen_bool(0.10) { 4 } else { 2 };
        Some(TileSpawn::new(row as u8, col as u8, value))
    }

    /// Generates a standardized initial board with 2 randomly spawned tiles using a specific seed.
    pub fn initial_with_seed(seed: u64) -> Tzf8State {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state = Tzf8State::new_empty();

        for _ in 0..2 {
            let empty = state.empty_cells();
            if !empty.is_empty() {
                let idx = rng.gen_range(0..empty.len());
                let (r, c) = empty[idx];
                let val = if rng.gen_bool(0.10) { 4 } else { 2 };
                state.apply_spawn(TileSpawn::new(r as u8, c as u8, val));
            }
        }

        state
    }

    /// Steps a single directional action in-place on `s`.
    pub fn step_action(&self, s: &mut Tzf8State, action: Direction) -> StepOutcome<[f32; 1]> {
        let (changed, score_gain) = s.move_board(action);
        if !changed {
            s.ongoing = false;
            return StepOutcome::new([0.0], true);
        }

        s.score += score_gain;

        if let Some(spawn) = self.sample_spawn(s) {
            s.apply_spawn(spawn);
        }

        let terminated = !s.can_move();
        if terminated {
            s.ongoing = false;
        }

        StepOutcome::new([score_gain as f32], terminated)
    }

    /// Returns `true` if the state is terminal (no legal moves remaining or stopped).
    pub fn is_terminal(&self, s: &Tzf8State) -> bool {
        self.terminal(s)
    }
}

impl World for Tzf8World {
    type WorldState = Tzf8State;
    type Action = Direction;
    type Observation = Tzf8State;

    fn n_players(&self) -> usize {
        1
    }

    fn initial(&self) -> Self::WorldState {
        let mut state = Tzf8State::new_empty();
        for _ in 0..2 {
            if let Some(spawn) = self.sample_spawn(&state) {
                state.apply_spawn(spawn);
            }
        }
        state
    }

    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        ws.clone()
    }

    fn actions(&self, ws: &Self::WorldState, _player: usize, out: &mut Vec<Self::Action>) {
        ws.legal_directions(out);
    }

    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        if joint.is_empty() {
            return (vec![0.0], self.terminal(ws));
        }
        let outcome = self.step_action(ws, joint[0]);
        (vec![outcome.reward[0]], outcome.terminated)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        !ws.ongoing || !ws.can_move()
    }
}

impl mcts_traits::TurnBasedWorld for Tzf8World {
    type StepReward = [f32; 1];

    #[inline]
    fn current_player(&self, _ws: &Self::WorldState) -> usize {
        0
    }

    #[inline]
    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> StepOutcome<Self::StepReward> {
        self.step_action(ws, *action)
    }
}
