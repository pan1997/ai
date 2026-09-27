//! Discrete directed graph MDP for exact algorithmic verification and tabular convergence.
//!
//! Provides a configurable discrete graph environment [`GraphEnv`] capable of modeling
//! chain MDPs, gridworlds, and arbitrary finite Markov Decision Processes.

use mcts_traits::{AgentDynamics, AgentId, StepOutcome};

/// A transition from state `s` under action `a` in a [`GraphEnv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphTransition {
    /// Target successor state ID.
    pub next_state: u32,
    /// Immediate step reward.
    pub reward: f32,
    /// Whether the transition leads to an absorbing terminal state.
    pub is_terminal: bool,
}

/// Fully observable, finite discrete Markov Decision Process defined by an adjacency table.
#[derive(Debug, Clone)]
pub struct GraphEnv {
    /// Cardinality of discrete state space $|\mathcal{S}|$.
    pub num_states: usize,
    /// Cardinality of discrete action space $|\mathcal{A}|$.
    pub num_actions: usize,
    /// Initial starting state ID.
    pub initial_state: u32,
    /// Transition table indexed by `[state][action]`.
    pub transitions: Vec<Vec<Option<GraphTransition>>>,
    /// Set of terminal states where no further actions are available.
    pub terminal_states: Vec<bool>,
}

impl GraphEnv {
    /// Creates a new `GraphEnv` with `num_states` states and `num_actions` actions.
    pub fn new(num_states: usize, num_actions: usize, initial_state: u32) -> Self {
        assert!(num_states > 0);
        assert!(num_actions > 0);
        assert!((initial_state as usize) < num_states);

        Self {
            num_states,
            num_actions,
            initial_state,
            transitions: vec![vec![None; num_actions]; num_states],
            terminal_states: vec![false; num_states],
        }
    }

    /// Sets a transition for `(state, action)`.
    pub fn set_transition(
        &mut self,
        state: u32,
        action: u32,
        next_state: u32,
        reward: f32,
        is_terminal: bool,
    ) {
        let s = state as usize;
        let a = action as usize;
        let ns = next_state as usize;
        assert!(s < self.num_states, "state {s} out of bounds");
        assert!(a < self.num_actions, "action {a} out of bounds");
        assert!(ns < self.num_states, "next_state {ns} out of bounds");

        self.transitions[s][a] = Some(GraphTransition {
            next_state,
            reward,
            is_terminal,
        });

        if is_terminal {
            self.terminal_states[ns] = true;
        }
    }

    /// Marks a state as terminal.
    pub fn mark_terminal(&mut self, state: u32) {
        let s = state as usize;
        assert!(s < self.num_states);
        self.terminal_states[s] = true;
    }

    /// Constructs a standard $K$-state Chain MDP ($0 \to 1 \to \dots \to K-1$).
    ///
    /// - Actions: `0` (Stay), `1` (Advance right).
    /// - Moving right from state $K-2$ to $K-1$ delivers reward $+1.0$ and terminates.
    /// - All other actions deliver reward $0.0$.
    /// - Optimal policy: $\pi^*(a=1 \mid s) = 1.0$ for all non-terminal states.
    pub fn chain(length: usize) -> Self {
        assert!(length >= 2, "Chain MDP must have at least 2 states");
        let mut env = Self::new(length, 2, 0);

        for s in 0..(length - 1) {
            let s_u32 = s as u32;
            // Action 0: Stay (or reset to 0 if s == 0)
            env.set_transition(s_u32, 0, s_u32.saturating_sub(1), 0.0, false);

            // Action 1: Forward
            let next_s = (s + 1) as u32;
            let is_last = s == length - 2;
            let reward = if is_last { 1.0 } else { 0.0 };
            env.set_transition(s_u32, 1, next_s, reward, is_last);
        }

        env.mark_terminal((length - 1) as u32);
        env
    }

    /// Returns whether `s` is a terminal state.
    #[inline]
    pub fn is_terminal(&self, s: u32) -> bool {
        let s = s as usize;
        if s >= self.num_states {
            return true;
        }
        self.terminal_states[s]
    }

    /// Produces a 1-hot observation vector of length `num_states` for state `s`.
    pub fn observation_for(&self, s: u32) -> Vec<f32> {
        let mut obs = vec![0.0; self.num_states];
        let idx = s as usize;
        if idx < self.num_states {
            obs[idx] = 1.0;
        }
        obs
    }

    /// Produces a bitpacked action mask for state `s` aligned to 4 bytes.
    pub fn action_mask_for(&self, s: u32) -> Vec<u8> {
        let mask_bytes = ((self.num_actions + 31) / 32) * 4;
        let mut mask = vec![0u8; mask_bytes];

        if !self.is_terminal(s) {
            let s_idx = s as usize;
            for a in 0..self.num_actions {
                if self.transitions[s_idx][a].is_some() {
                    mask[a / 8] |= 1 << (a % 8);
                }
            }
        }
        mask
    }
}

impl AgentDynamics for GraphEnv {
    type State = u32;
    type Action = u32;
    type Reward = [f32; 1];
    type StepDelta = u32;

    fn initial(&self) -> Self::State {
        self.initial_state
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        let idx = *s as usize;
        if idx >= self.num_states || self.terminal_states[idx] {
            return;
        }

        for a in 0..self.num_actions {
            if self.transitions[idx][a].is_some() {
                out.push(a as u32);
            }
        }
    }

    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> StepOutcome<Self::Reward, Self::StepDelta> {
        let s_idx = *s as usize;
        let a_idx = *action as usize;

        let trans = self.transitions[s_idx][a_idx]
            .unwrap_or_else(|| panic!("GraphEnv: illegal action {a_idx} from state {s_idx}"));

        *s = trans.next_state;
        StepOutcome::with_delta([trans.reward], trans.next_state, trans.is_terminal)
    }

    fn current_agent(&self, _s: &Self::State) -> AgentId {
        AgentId(0)
    }
}
