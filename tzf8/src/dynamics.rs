//! Single-agent transition dynamics and stochastic chance branching for 2048 MCTS planning.

use crate::game::{Direction, TileSpawn, Tzf8State};
use mcts_traits::{AgentDynamics, AgentId, StepOutcome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

/// Planning transition dynamics for 2048.
///
/// Models hypothetically sliding tiles on a board, earning immediate merge rewards,
/// and sampling stochastic chance tile spawns (`TileSpawn`) into `StepDelta`.
///
/// Because tile spawn events are recorded into `StepDelta`, the MCTS search tree
/// forms delta branches under each edge:
///
/// $$Q(s, a) = r(s, a) + \gamma \sum_{\text{spawn}} \mathbb{P}(\text{spawn} \mid s, a) V(s')$$
///
/// As multiple simulation sweeps explore edge $(s, a)$, different random tile spawns
/// branch out as distinct child nodes, converging naturally to true Expectimax values.
pub struct Tzf8Dynamics {
    rng: Mutex<StdRng>,
}

impl Default for Tzf8Dynamics {
    fn default() -> Self {
        Self::new()
    }
}

impl Tzf8Dynamics {
    /// Creates a new `Tzf8Dynamics` seeded from system entropy.
    pub fn new() -> Self {
        Self {
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Creates a new `Tzf8Dynamics` initialized with a deterministic seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Samples a stochastic tile spawn on `state` (90% probability 2, 10% probability 4).
    pub fn sample_spawn(&self, state: &Tzf8State) -> Option<TileSpawn> {
        let empty = state.empty_cells();
        if empty.is_empty() {
            return None;
        }
        let mut rng = self.rng.lock().unwrap();
        let idx = rng.gen_range(0..empty.len());
        let (row, col) = empty[idx];
        let value = if rng.gen_bool(0.10) { 4 } else { 2 };
        Some(TileSpawn::new(row as u8, col as u8, value))
    }
}

impl AgentDynamics for Tzf8Dynamics {
    type State = Tzf8State;
    type Action = Direction;
    type Reward = [f32; 1];
    type StepDelta = Option<TileSpawn>;

    /// Returns a standardized fresh board containing two initial random tiles.
    fn initial(&self) -> Self::State {
        let mut state = Tzf8State::new_empty();
        for _ in 0..2 {
            if let Some(spawn) = self.sample_spawn(&state) {
                state.apply_spawn(spawn);
            }
        }
        state
    }

    /// Populates `out` with all legal directional slides from the current state.
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        s.legal_directions(out);
    }

    /// Advances `s` in-place by sliding in direction `action`, accumulating merge score,
    /// and sampling a stochastic tile spawn.
    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        let (changed, score_gain) = s.move_board(*action);
        if !changed {
            s.ongoing = false;
            return StepOutcome::with_delta([0.0], None, true);
        }

        s.score += score_gain;

        let spawn = self.sample_spawn(s);
        if let Some(sp) = spawn {
            s.apply_spawn(sp);
        }

        let terminated = !s.can_move();
        if terminated {
            s.ongoing = false;
        }

        StepOutcome::with_delta([score_gain as f32], spawn, terminated)
    }

    #[inline]
    fn current_agent(&self, _s: &Self::State) -> AgentId {
        AgentId(0)
    }
}
