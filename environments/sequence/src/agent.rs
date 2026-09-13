//! Agent implementations for Sequence: Human CLI, Random, Heuristic, ISMCTS, Opponent-Model MCTS.

use crate::dynamics::{
    determinize_state, RandomOpponentPolicy, SequenceRoundDynamics, SequenceTurnDynamics,
};
use crate::evaluator::{SequenceHeuristicEvaluator, UniformEvaluator};
use crate::game::{SequenceAction, SequenceState};
use crate::render::{format_action, render_state};
use crate::world::SequenceWorld;
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::dynamics::OpponentPolicy;
use mcts_traits::{Agent, AgentId, Model, World};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// Dynamic boxed Sequence agent.
pub type BoxAgent<const P: usize = 2> = Box<dyn Agent<SequenceState, SequenceAction>>;

/// Interactive human player choosing actions via terminal stdin.
pub struct HumanAgent {
    name: String,
}

impl HumanAgent {
    /// Creates a new human player with the specified name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Agent<SequenceState, SequenceAction> for HumanAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("HumanAgent: no legal actions available");
        }
        if legal.len() == 1 {
            return legal[0];
        }

        println!("{}", render_state(state, true));
        println!("\n=== {}'s Turn (Player {}) ===", self.name, state.current_player);
        println!("Available legal moves ({} total):", legal.len());
        for (i, action) in legal.iter().take(20).enumerate() {
            println!("  [{:2}] {}", i, format_action(action));
        }
        if legal.len() > 20 {
            println!("  ... and {} more options. (Type 'list' to see all)", legal.len() - 20);
        }

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("Choose move index [0..{}]: ", legal.len() - 1);
            let _ = stdout.flush();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).is_err() {
                println!("Error reading input, please try again.");
                continue;
            }

            let trimmed = input.trim();
            if trimmed.eq_ignore_ascii_case("list") {
                for (i, action) in legal.iter().enumerate() {
                    println!("  [{:3}] {}", i, format_action(action));
                }
                continue;
            }

            match trimmed.parse::<usize>() {
                Ok(idx) if idx < legal.len() => return legal[idx],
                _ => println!("Invalid index. Enter a number between 0 and {}.", legal.len() - 1),
            }
        }
    }
}

/// Agent selecting uniform random legal actions.
pub struct RandomAgent {
    name: String,
}

impl RandomAgent {
    /// Creates a new random agent.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Agent<SequenceState, SequenceAction> for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("RandomAgent: no legal actions available");
        }
        let mut rng = rand::thread_rng();
        *legal.choose(&mut rng).expect("non-empty legal actions")
    }
}

/// 1-ply greedy heuristic agent selecting the highest-scoring immediate action.
pub struct HeuristicAgent {
    name: String,
    evaluator: SequenceHeuristicEvaluator,
}

impl HeuristicAgent {
    /// Creates a new heuristic agent.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            evaluator: SequenceHeuristicEvaluator::new(),
        }
    }
}

impl Agent<SequenceState, SequenceAction> for HeuristicAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("HeuristicAgent: no legal actions available");
        }
        if legal.len() == 1 {
            return legal[0];
        }

        let mut best_action = legal[0];
        let mut best_score = f32::NEG_INFINITY;

        for action in &legal {
            let score = self.evaluator.score_action(state, action);
            if score > best_score {
                best_score = score;
                best_action = *action;
            }
        }

        best_action
    }
}

/// Information Set MCTS (ISMCTS) agent.
///
/// Under imperfect information, pools unseen cards and generates $D$ independent random
/// root determinizations. Runs $I / D$ MCTS iterations on each determinization and aggregates
/// root visit counts to make robust decisions without cheating.
pub struct IsMctsAgent<M, const P: usize = 2> {
    name: String,
    total_iterations: usize,
    num_determinizations: usize,
    c_puct: f32,
    model: M,
}

impl<const P: usize> IsMctsAgent<SequenceHeuristicEvaluator, P> {
    /// Creates a new ISMCTS agent guided by the heuristic evaluator.
    #[must_use]
    pub fn new_heuristic(
        name: impl Into<String>,
        total_iterations: usize,
        num_determinizations: usize,
    ) -> Self {
        Self {
            name: name.into(),
            total_iterations,
            num_determinizations: num_determinizations.max(1),
            c_puct: 1.414,
            model: SequenceHeuristicEvaluator::new(),
        }
    }
}

impl<const P: usize> IsMctsAgent<UniformEvaluator, P> {
    /// Creates a new ISMCTS agent with uniform priors.
    #[must_use]
    pub fn new_uniform(
        name: impl Into<String>,
        total_iterations: usize,
        num_determinizations: usize,
    ) -> Self {
        Self {
            name: name.into(),
            total_iterations,
            num_determinizations: num_determinizations.max(1),
            c_puct: 1.414,
            model: UniformEvaluator,
        }
    }
}

impl<M, const P: usize> Agent<SequenceState, SequenceAction> for IsMctsAgent<M, P>
where
    M: Model<SequenceState> + Clone,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("IsMctsAgent: no legal actions available");
        }
        if legal.len() == 1 {
            return legal[0];
        }

        let world = SequenceWorld::<P>::new(state.config);
        let obs = world.observe(state, state.current_player);

        let iters_per_det = (self.total_iterations / self.num_determinizations).max(1);
        let mut aggregated_visits: HashMap<SequenceAction, u32> = HashMap::new();

        let mut rng = rand::thread_rng();
        for _ in 0..self.num_determinizations {
            let hyp_state = determinize_state(&obs, &mut rng);

            let dynamics = SequenceTurnDynamics::new(world);
            let selection = MultiAgentPuctSelection::<P> { c_puct: self.c_puct };
            let backup = VectorBackup::<P>::default();
            let stats = MultiAgentPuctStats::<P>::new();

            let node_cap = iters_per_det + 16;
            let edge_cap = node_cap * 32;
            let mut tree = TreeStore::with_capacity(node_cap, edge_cap, stats);
            let root = tree.insert_root(AgentId(hyp_state.current_player as u32));

            let scheduler = SequentialScheduler;
            scheduler.search(
                &mut tree,
                &dynamics,
                &self.model,
                &selection,
                &backup,
                root,
                &hyp_state,
                iters_per_det,
            );

            let num_children = tree.num_children(root);
            let first_edge = tree.first_child_edge(root);
            for i in 0..num_children {
                let edge = mcts_engine::tree_store::EdgeId(first_edge.0 + i);
                let edge_idx = edge.as_usize();
                let action = *tree.edge_action(edge);
                let visits = tree.stats.visits[edge_idx];
                *aggregated_visits.entry(action).or_insert(0) += visits;
            }
        }

        // Return legal action with maximum total visits
        aggregated_visits
            .into_iter()
            .max_by_key(|&(_, visits)| visits)
            .map(|(action, _)| action)
            .unwrap_or(legal[0])
    }
}

/// Standard MCTS agent running search on the ground-truth state (perfect-information oracle).
pub struct MctsAgent<M, const P: usize = 2> {
    name: String,
    num_iterations: usize,
    c_puct: f32,
    model: M,
}

impl<const P: usize> MctsAgent<SequenceHeuristicEvaluator, P> {
    /// Creates a heuristic-guided MCTS agent.
    #[must_use]
    pub fn new_heuristic(name: impl Into<String>, num_iterations: usize) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct: 1.414,
            model: SequenceHeuristicEvaluator::new(),
        }
    }
}

impl<const P: usize> MctsAgent<UniformEvaluator, P> {
    /// Creates an MCTS agent with uniform evaluation.
    #[must_use]
    pub fn new_uniform(name: impl Into<String>, num_iterations: usize) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct: 1.414,
            model: UniformEvaluator,
        }
    }
}

impl<M, const P: usize> Agent<SequenceState, SequenceAction> for MctsAgent<M, P>
where
    M: Model<SequenceState>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("MctsAgent: no legal actions available");
        }
        if legal.len() == 1 {
            return legal[0];
        }

        let world = SequenceWorld::<P>::new(state.config);
        let dynamics = SequenceTurnDynamics::new(world);
        let selection = MultiAgentPuctSelection::<P> { c_puct: self.c_puct };
        let backup = VectorBackup::<P>::default();
        let stats = MultiAgentPuctStats::<P>::new();

        let node_cap = self.num_iterations + 16;
        let edge_cap = node_cap * 32;
        let mut tree = TreeStore::with_capacity(node_cap, edge_cap, stats);
        let root = tree.insert_root(AgentId(state.current_player as u32));

        let sim_state = state.clone();
        let scheduler = SequentialScheduler;
        scheduler.search(
            &mut tree,
            &dynamics,
            &self.model,
            &selection,
            &backup,
            root,
            &sim_state,
            self.num_iterations,
        );

        let num_children = tree.num_children(root);
        let first_edge = tree.first_child_edge(root);
        let mut best_action = legal[0];
        let mut max_visits = 0;

        for i in 0..num_children {
            let edge = mcts_engine::tree_store::EdgeId(first_edge.0 + i);
            let edge_idx = edge.as_usize();
            let visits = tree.stats.visits[edge_idx];
            if visits > max_visits {
                max_visits = visits;
                best_action = *tree.edge_action(edge);
            }
        }

        best_action
    }
}

/// Heuristic opponent policy for round-based macro simulations.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicOpponentPolicy {
    evaluator: SequenceHeuristicEvaluator,
}

impl OpponentPolicy<SequenceState, SequenceAction> for HeuristicOpponentPolicy {
    fn select_action(&self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("HeuristicOpponentPolicy: no legal actions available");
        }
        let mut best_action = legal[0];
        let mut best_score = f32::NEG_INFINITY;
        for action in &legal {
            let score = self.evaluator.score_action(state, action);
            if score > best_score {
                best_score = score;
                best_action = *action;
            }
        }
        best_action
    }
}

/// Opponent-Model MCTS Agent: uses random root determinization and round-based macro dynamics
/// with an explicit opponent policy model (heuristic or random).
pub struct OpponentModelMctsAgent<Pol, const P: usize = 2> {
    name: String,
    total_iterations: usize,
    num_determinizations: usize,
    c_puct: f32,
    opponent_policy: Pol,
    evaluator: SequenceHeuristicEvaluator,
}

impl<const P: usize> OpponentModelMctsAgent<HeuristicOpponentPolicy, P> {
    /// Creates a macro-dynamics MCTS agent assuming heuristic opponent behavior.
    #[must_use]
    pub fn new_heuristic(
        name: impl Into<String>,
        total_iterations: usize,
        num_determinizations: usize,
    ) -> Self {
        Self {
            name: name.into(),
            total_iterations,
            num_determinizations: num_determinizations.max(1),
            c_puct: 1.414,
            opponent_policy: HeuristicOpponentPolicy::default(),
            evaluator: SequenceHeuristicEvaluator::new(),
        }
    }
}

impl<const P: usize> OpponentModelMctsAgent<RandomOpponentPolicy, P> {
    /// Creates a macro-dynamics MCTS agent assuming uniform random opponent behavior.
    #[must_use]
    pub fn new_random(
        name: impl Into<String>,
        total_iterations: usize,
        num_determinizations: usize,
    ) -> Self {
        Self {
            name: name.into(),
            total_iterations,
            num_determinizations: num_determinizations.max(1),
            c_puct: 1.414,
            opponent_policy: RandomOpponentPolicy,
            evaluator: SequenceHeuristicEvaluator::new(),
        }
    }
}

impl<Pol, const P: usize> Agent<SequenceState, SequenceAction> for OpponentModelMctsAgent<Pol, P>
where
    Pol: OpponentPolicy<SequenceState, SequenceAction> + Clone,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &SequenceState) -> SequenceAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("OpponentModelMctsAgent: no legal actions available");
        }
        if legal.len() == 1 {
            return legal[0];
        }

        let world = SequenceWorld::<P>::new(state.config);
        let obs = world.observe(state, state.current_player);

        let iters_per_det = (self.total_iterations / self.num_determinizations).max(1);
        let mut aggregated_visits: HashMap<SequenceAction, u32> = HashMap::new();

        let mut rng = rand::thread_rng();
        for _ in 0..self.num_determinizations {
            let hyp_state = determinize_state(&obs, &mut rng);

            let dynamics = SequenceRoundDynamics::new(
                world,
                self.opponent_policy.clone(),
                state.current_player,
            );
            let selection = MultiAgentPuctSelection::<P> { c_puct: self.c_puct };
            let backup = VectorBackup::<P>::default();
            let stats = MultiAgentPuctStats::<P>::new();

            let node_cap = iters_per_det + 16;
            let edge_cap = node_cap * 32;
            let mut tree = TreeStore::with_capacity(node_cap, edge_cap, stats);
            let root = tree.insert_root(AgentId(hyp_state.current_player as u32));

            let scheduler = SequentialScheduler;
            scheduler.search(
                &mut tree,
                &dynamics,
                &self.evaluator,
                &selection,
                &backup,
                root,
                &hyp_state,
                iters_per_det,
            );

            let num_children = tree.num_children(root);
            let first_edge = tree.first_child_edge(root);
            for i in 0..num_children {
                let edge = mcts_engine::tree_store::EdgeId(first_edge.0 + i);
                let edge_idx = edge.as_usize();
                let action = *tree.edge_action(edge);
                let visits = tree.stats.visits[edge_idx];
                *aggregated_visits.entry(action).or_insert(0) += visits;
            }
        }

        aggregated_visits
            .into_iter()
            .max_by_key(|&(_, visits)| visits)
            .map(|(action, _)| action)
            .unwrap_or(legal[0])
    }
}

/// Builds a boxed Sequence agent from a specification string.
///
/// Supported specs:
/// - `"human"`
/// - `"random"`
/// - `"heuristic"`
/// - `"mcts:<iters>"` (e.g. `"mcts:200"`)
/// - `"is-mcts:<iters>:<dets>"` (e.g. `"is-mcts:300:5"`)
/// - `"macro-heuristic:<iters>:<dets>"` (e.g. `"macro-heuristic:300:5"`)
/// - `"macro-random:<iters>:<dets>"` (e.g. `"macro-random:300:5"`)
pub fn parse_agent<const P: usize>(spec: &str, default_name: &str) -> BoxAgent<P> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts[0].to_lowercase().as_str() {
        "human" => Box::new(HumanAgent::new(default_name)),
        "random" => Box::new(RandomAgent::new(default_name)),
        "heuristic" => Box::new(HeuristicAgent::new(default_name)),
        "mcts" => {
            let iters = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
            Box::new(MctsAgent::<SequenceHeuristicEvaluator, P>::new_heuristic(default_name, iters))
        }
        "is-mcts" => {
            let iters = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
            let dets = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            Box::new(IsMctsAgent::<SequenceHeuristicEvaluator, P>::new_heuristic(default_name, iters, dets))
        }
        "macro-heuristic" => {
            let iters = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
            let dets = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            Box::new(OpponentModelMctsAgent::<HeuristicOpponentPolicy, P>::new_heuristic(default_name, iters, dets))
        }
        "macro-random" => {
            let iters = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
            let dets = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            Box::new(OpponentModelMctsAgent::<RandomOpponentPolicy, P>::new_random(default_name, iters, dets))
        }
        other => panic!("Unknown agent specification: '{other}'. Expected human, random, heuristic, mcts:<iters>, is-mcts:<iters>:<dets>, macro-heuristic:<iters>:<dets>, or macro-random:<iters>:<dets>"),
    }
}
