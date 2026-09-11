use crate::agent::AgentId;
use crate::dynamics::{AgentDynamics, BatchedAgentDynamics, Transition};
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
    pub fn new(initial_state: u32) -> Self {
        Self {
            initial_state,
            transitions: HashMap::new(),
            legal_actions: HashMap::new(),
            state_agents: HashMap::new(),
            terminal_states: HashSet::new(),
        }
    }

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

    pub fn set_actions(&mut self, state: u32, actions: Vec<u32>) {
        self.legal_actions.insert(state, actions);
    }

    pub fn set_agent(&mut self, state: u32, agent: AgentId) {
        self.state_agents.insert(state, agent);
    }

    pub fn set_terminal(&mut self, state: u32, terminated: bool) {
        if terminated {
            self.terminal_states.insert(state);
        } else {
            self.terminal_states.remove(&state);
        }
    }

    pub fn current_agent(&self, state: &u32) -> AgentId {
        *self.state_agents.get(state).unwrap_or(&AgentId(0))
    }

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

    fn actions(&self, s: &Self::State) -> Vec<Self::Action> {
        self.legal_actions.get(s).cloned().unwrap_or_default()
    }

    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward> {
        self.transitions
            .get(&(s, *action))
            .copied()
            .unwrap_or_else(|| panic!("GraphEnv: invalid transition queried for state {s}, action {action}"))
    }
}

impl<const N: usize> BatchedAgentDynamics for GraphEnv<N> {
    fn step_batch(
        &self,
        states: &[Self::State],
        actions: &[Self::Action],
        out_transitions: &mut Vec<Transition<Self::State, Self::Reward>>,
    ) {
        crate::dynamics::default_step_batch(self, states, actions, out_transitions);
    }
}

