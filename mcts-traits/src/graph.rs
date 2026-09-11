use crate::agent::AgentId;
use crate::dynamics::{AgentDynamics, BatchedAgentDynamics, StepOutcome, Transition};
use std::collections::{HashMap, HashSet};

/// Configurable, deterministic state-machine graph for exact mathematical unit tests.
///
/// Designed to test MCTS engine mechanics (PUCT, UCT, Gumbel, virtual loss, vector backup)
/// on hand-calculable graphs where ground truth is known precisely.
#[derive(Debug, Clone)]
pub struct GraphEnv<const N: usize = 1> {
    pub initial_state: u32,
    pub transitions: HashMap<(u32, u32), Transition<u32, [f32; N]>>,
    pub legal_actions: HashMap<u32, Vec<u32>>,
    pub state_agents: HashMap<u32, AgentId>,
    pub terminal_states: HashSet<u32>,
}

impl<const N: usize> GraphEnv<N> {
    /// Creates a new `GraphEnv` with the designated initial state index.
    pub fn new(initial_state: u32) -> Self {
        Self {
            initial_state,
            transitions: HashMap::new(),
            legal_actions: HashMap::new(),
            state_agents: HashMap::new(),
            terminal_states: HashSet::new(),
        }
    }

    /// Inserts a deterministic directed transition `(from_state, action) -> (next_state, reward, terminated)`.
    pub fn add_transition(
        &mut self,
        from_state: u32,
        action: u32,
        next_state: u32,
        reward: [f32; N],
        terminated: bool,
    ) {
        self.transitions.insert(
            (from_state, action),
            Transition::new(next_state, reward, terminated),
        );
        if terminated {
            self.terminal_states.insert(next_state);
        }
    }

    /// Sets the list of legal action IDs available from `state`.
    pub fn set_actions(&mut self, state: u32, actions: Vec<u32>) {
        self.legal_actions.insert(state, actions);
    }

    /// Sets which agent is acting at `state`.
    pub fn set_agent(&mut self, state: u32, agent: AgentId) {
        self.state_agents.insert(state, agent);
    }

    /// Marks or unmarks `state` as terminal.
    pub fn set_terminal(&mut self, state: u32, terminated: bool) {
        if terminated {
            self.terminal_states.insert(state);
        } else {
            self.terminal_states.remove(&state);
        }
    }

    /// Returns the agent active at `state` (defaults to `AgentId(0)` if not set).
    pub fn current_agent(&self, state: &u32) -> AgentId {
        *self.state_agents.get(state).unwrap_or(&AgentId(0))
    }

    /// Returns `true` if `state` is a terminal state.
    pub fn is_terminal(&self, state: &u32) -> bool {
        self.terminal_states.contains(state)
    }
}

impl<const N: usize> AgentDynamics for GraphEnv<N> {
    type State = u32;
    type Action = u32;
    type Reward = [f32; N];

    fn initial(&self) -> Self::State {
        self.initial_state
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        if let Some(actions) = self.legal_actions.get(s) {
            out.extend_from_slice(actions);
        }
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward> {
        let t = self
            .transitions
            .get(&(*s, *action))
            .copied()
            .unwrap_or_else(|| panic!("GraphEnv: invalid transition queried for state {s}, action {action}"));
        *s = t.next_state;
        StepOutcome::new(t.reward, t.terminated)
    }
}

impl<const N: usize> BatchedAgentDynamics for GraphEnv<N> {
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        out_outcomes: &mut Vec<StepOutcome<Self::Reward>>,
    ) {
        crate::dynamics::default_step_batch(self, states, actions, out_outcomes);
    }
}

