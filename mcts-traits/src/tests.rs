use crate::agent::AgentId;
use crate::dynamics::{
    AgentDynamics, OpponentPolicy, RoundBasedDynamics, StepOutcome, Transition, TurnBasedDynamics,
    default_step_batch,
};
use crate::graph::GraphEnv;
use crate::model::{Evaluation, HasPolicy, HasValue};
use crate::world::{TurnBasedWorld, World};

#[test]
fn test_agent_id_conversions_and_display() {
    let a0 = AgentId::new(0);
    let a5 = AgentId::new(5);

    assert_eq!(a0.as_u32(), 0);
    assert_eq!(a0.as_usize(), 0);
    assert_eq!(a5.as_u32(), 5);
    assert_eq!(a5.as_usize(), 5);

    // Conversions
    assert_eq!(AgentId::from(10u32), AgentId(10));
    assert_eq!(AgentId::from(7usize), AgentId(7));
    let idx: usize = a5.into();
    assert_eq!(idx, 5);

    // Default
    assert_eq!(AgentId::default(), AgentId(0));

    // Display
    assert_eq!(format!("{a5}"), "Agent(5)");

    // Ordering and Equality
    assert!(a0 < a5);
    assert_eq!(a0, AgentId(0));
}

#[test]
fn test_transition_struct() {
    let t = Transition::new(10u32, 1.5f32, false);
    assert_eq!(t.next_state, 10);
    assert_eq!(t.reward, 1.5);
    assert!(!t.terminated);

    let t2 = t; // Copy
    assert_eq!(t, t2);
}

#[test]
fn test_step_outcome_struct() {
    let outcome = StepOutcome::new(1.5f32, false);
    assert_eq!(outcome.reward, 1.5);
    assert!(!outcome.terminated);

    let o2 = outcome;
    assert_eq!(outcome, o2);
}

#[test]
fn test_evaluation_and_traits() {
    // Scalar evaluation
    let eval_scalar = Evaluation::scalar(vec![0.7, 0.3], 0.85);
    assert_eq!(eval_scalar.priors(), &[0.7, 0.3]);
    assert_eq!(eval_scalar.values(), &[0.85]);
    assert_eq!(eval_scalar.value(), 0.85);

    // Vector evaluation
    let eval_vec = Evaluation::vector(vec![0.5, 0.5], vec![1.0, -1.0]);
    assert_eq!(eval_vec.priors(), &[0.5, 0.5]);
    assert_eq!(eval_vec.values(), &[1.0, -1.0]);
    assert_eq!(eval_vec.value(), 1.0); // Returns first agent value

    // Empty values fallback
    let eval_empty = Evaluation {
        priors: vec![],
        values: vec![],
    };
    assert_eq!(eval_empty.value(), 0.0);
    assert!(eval_empty.values().is_empty());
    assert!(eval_empty.priors().is_empty());
}

#[derive(Clone)]
struct DummyDynamics;

impl AgentDynamics for DummyDynamics {
    type State = u32;
    type Action = u32;
    type Reward = f32;
    type StepDelta = ();

    fn initial(&self) -> Self::State {
        0
    }

    fn actions(&self, _s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        out.extend([1, 2]);
    }

    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward, ()> {
        *s += *action;
        StepOutcome::new(*action as f32 * 10.0, false)
    }
}

#[test]
fn test_default_step_batch() {
    let dyns = DummyDynamics;
    let mut states = [10, 20, 30];
    let actions = [1, 2, 1];
    let mut outcomes = Vec::new();

    default_step_batch(&dyns, &mut states, &actions, &mut outcomes);

    assert_eq!(outcomes.len(), 3);
    assert_eq!(states[0], 11);
    assert_eq!(outcomes[0].reward, 10.0);
    assert_eq!(states[1], 22);
    assert_eq!(outcomes[1].reward, 20.0);
    assert_eq!(states[2], 31);
    assert_eq!(outcomes[2].reward, 10.0);
}

#[test]
#[should_panic(expected = "default_step_batch: states and actions slice lengths must match")]
fn test_default_step_batch_mismatched_lengths_panic() {
    let dyns = DummyDynamics;
    let mut states = [10, 20];
    let actions = [1];
    let mut outcomes = Vec::new();

    default_step_batch(&dyns, &mut states, &actions, &mut outcomes);
}

#[test]
fn test_graph_env_lifecycle_and_defaults() {
    let mut env = GraphEnv::<2>::new(0);

    // Initial state
    assert_eq!(env.initial(), 0);
    assert!(!env.is_terminal(&0));
    assert_eq!(env.current_agent(&0), AgentId(0));
    let mut acts = Vec::new();
    env.actions(&0, &mut acts);
    assert!(acts.is_empty());

    // Configure transitions
    env.set_actions(0, vec![1, 2]);
    env.set_agent(0, AgentId(1));
    env.add_transition(0, 1, 10, [1.0, -1.0], false);
    env.add_transition(0, 2, 20, [0.0, 0.0], true);

    env.actions(&0, &mut acts);
    assert_eq!(acts, vec![1, 2]);
    assert_eq!(env.current_agent(&0), AgentId(1));

    let mut s1 = 0;
    let t1 = env.step(&mut s1, &1);
    assert_eq!(s1, 10);
    assert_eq!(t1.reward, [1.0, -1.0]);
    assert!(!t1.terminated);

    let mut s2 = 0;
    let t2 = env.step(&mut s2, &2);
    assert_eq!(s2, 20);
    assert!(t2.terminated);
    assert!(env.is_terminal(&20));

    // Change terminal status
    env.set_terminal(20, false);
    assert!(!env.is_terminal(&20));
}

#[test]
#[should_panic(expected = "GraphEnv: invalid transition queried for state 99, action 99")]
fn test_graph_env_invalid_transition_panic() {
    let env = GraphEnv::<1>::new(0);
    let mut s = 99;
    env.step(&mut s, &99);
}

struct MockTurnBasedWorld;

impl World for MockTurnBasedWorld {
    type WorldState = (usize, usize);
    type Action = usize;
    type Observation = (usize, usize);

    fn n_players(&self) -> usize {
        2
    }
    fn initial(&self) -> Self::WorldState {
        (0, 0)
    }
    fn observe(&self, ws: &Self::WorldState, _player: usize) -> Self::Observation {
        *ws
    }
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        if player == ws.0 % 2 {
            out.extend([1, 2]);
        }
    }
    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        let player = ws.0 % 2;
        let outcome = self.step_action(ws, &joint[player]);
        (outcome.reward.to_vec(), outcome.terminated)
    }
    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.0 >= 4
    }
}

impl TurnBasedWorld for MockTurnBasedWorld {
    type StepReward = [f32; 2];

    fn current_player(&self, ws: &Self::WorldState) -> usize {
        ws.0 % 2
    }

    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> StepOutcome<[f32; 2]> {
        ws.1 += *action;
        ws.0 += 1;
        let term = ws.0 >= 4;
        StepOutcome::new(if term { [1.0, -1.0] } else { [0.0, 0.0] }, term)
    }
}

#[test]
fn test_turn_based_dynamics_adapter() {
    let world = MockTurnBasedWorld;
    let dyns = TurnBasedDynamics::new(world);

    let mut state = dyns.initial();
    assert_eq!(state, (0, 0));
    assert_eq!(dyns.current_agent(&state), AgentId(0));

    let mut acts = Vec::new();
    dyns.actions(&state, &mut acts);
    assert_eq!(acts, vec![1, 2]);

    let outcome1 = dyns.step(&mut state, &1);
    assert!(!outcome1.terminated);
    assert_eq!(state, (1, 1));
    assert_eq!(dyns.current_agent(&state), AgentId(1));

    let outcome2 = dyns.step(&mut state, &2);
    assert!(!outcome2.terminated);
    assert_eq!(state, (2, 3));
    assert_eq!(dyns.current_agent(&state), AgentId(0));
}

struct FixedOpponent(usize);

impl OpponentPolicy<(usize, usize), usize> for FixedOpponent {
    fn select_action(&self, _state: &(usize, usize)) -> usize {
        self.0
    }
}

#[test]
fn test_round_based_dynamics_adapter() {
    let world = MockTurnBasedWorld;
    let opponent = FixedOpponent(10);
    let dyns = RoundBasedDynamics::new(world, opponent, 0);

    let mut state = dyns.initial();
    assert_eq!(state, (0, 0));
    assert_eq!(dyns.current_agent(&state), AgentId(0));

    // Step primary action 2:
    // Primary plays 2 (ws.0 = 1, ws.1 = 2).
    // Opponent is player 1 != 0, plays fixed 10 (ws.0 = 2, ws.1 = 12).
    // Now ws.0 = 2 (player 0's turn again!). Round completes!
    let outcome = dyns.step(&mut state, &2);
    assert!(!outcome.terminated);
    assert_eq!(outcome.delta, Some(10));
    assert_eq!(state, (2, 12));
    assert_eq!(dyns.current_agent(&state), AgentId(0));

    // Next round: primary plays 3 (ws.0 = 3, ws.1 = 15).
    // Opponent plays 10 (ws.0 = 4, ws.1 = 25).
    // Terminal condition ws.0 >= 4 reached!
    let outcome2 = dyns.step(&mut state, &3);
    assert!(outcome2.terminated);
    assert_eq!(outcome2.delta, Some(10));
    assert_eq!(outcome2.reward, [1.0, -1.0]);
}
