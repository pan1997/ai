//! Connect 4 Agent abstractions, Human CLI interaction, Random, and MCTS player implementations.

use crate::dynamics::TacticalOpponent;
use crate::evaluator::{RolloutEvaluator, UniformEvaluator};
use crate::game::Connect4State;
use crate::render::{MoveCandidate, format_move_candidates};
use crate::world::Connect4World;
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentId, Model, TurnBasedDynamics};
use rand::seq::SliceRandom;
use std::io::{self, BufRead, Write};

pub use mcts_traits::Agent;

/// Dynamic boxed Connect 4 agent.
pub type BoxAgent<const R: usize = 6, const C: usize = 7> =
    Box<dyn Agent<Connect4State<R, C>, usize>>;

/// Interactive human player prompting for moves via standard input.
pub struct HumanAgent {
    name: String,
}

impl HumanAgent {
    /// Creates a new human player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Returns the player name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<const R: usize, const C: usize> mcts_traits::Agent<Connect4State<R, C>, usize> for HumanAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Connect4State<R, C>) -> usize {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);

        loop {
            print!(
                "{} ({:?}), enter column {:?}: ",
                self.name, state.current_player, legal
            );
            let _ = stdout.flush();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).is_err() {
                println!("Error reading input, please try again.");
                continue;
            }

            let trimmed = input.trim();
            match trimmed.parse::<usize>() {
                Ok(col) => {
                    if col >= C {
                        println!("Column {col} is out of bounds (valid range: 0..{C}).");
                    } else if state.is_column_full(col) {
                        println!("Column {col} is full! Choose another column.");
                    } else {
                        return col;
                    }
                }
                Err(_) => {
                    println!("Invalid input '{trimmed}'. Please enter an integer column number.");
                }
            }
        }
    }
}

/// Baseline agent selecting uniformly at random among legal actions.
pub struct RandomAgent {
    name: String,
}

impl RandomAgent {
    /// Creates a new random player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Returns the player name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self::new("Random Agent")
    }
}

impl<const R: usize, const C: usize> mcts_traits::Agent<Connect4State<R, C>, usize>
    for RandomAgent
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Connect4State<R, C>) -> usize {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        let mut rng = rand::thread_rng();
        *legal
            .choose(&mut rng)
            .expect("RandomAgent: no legal actions available in state")
    }
}

/// Tactical heuristic agent: plays immediate winning move, blocks opponent 1-ply win, or center preference.
pub struct TacticalAgent {
    name: String,
    opponent: TacticalOpponent,
}

impl TacticalAgent {
    /// Creates a new tactical agent with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            opponent: TacticalOpponent,
        }
    }
}

impl Default for TacticalAgent {
    fn default() -> Self {
        Self::new("Tactical Agent")
    }
}

impl<const R: usize, const C: usize> mcts_traits::Agent<Connect4State<R, C>, usize>
    for TacticalAgent
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Connect4State<R, C>) -> usize {
        crate::dynamics::OpponentPolicy::select_action(&self.opponent, state)
    }
}

///// Opponent modeling mode for Connect 4 MCTS planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MctsMode {
    /// Full-tree 1-ply alternating adversarial MCTS (standard zero-sum minimax tree).
    Adversarial,
    /// Macro-action MCTS assuming opponent follows a tactical heuristic.
    MacroTactical,
    /// Macro-action MCTS assuming opponent chooses uniform random moves.
    MacroRandom,
}

/// Monte Carlo Tree Search agent for Connect 4 supporting multiple opponent modeling modes.
pub struct MctsAgent<M, const R: usize = 6, const C: usize = 7> {
    name: String,
    /// Number of MCTS simulation sweeps.
    pub num_iterations: usize,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Evaluation model.
    pub model: M,
    /// Opponent modeling mode (Adversarial, MacroTactical, MacroRandom).
    pub mode: MctsMode,
    /// Whether to print candidate move statistics to standard output.
    pub verbose: bool,
}

impl<M, const R: usize, const C: usize> MctsAgent<M, R, C> {
    /// Creates a new MCTS agent with custom evaluation model, parameters, and mode.
    pub fn new_with_mode(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        mode: MctsMode,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            model,
            mode,
            verbose,
        }
    }

    /// Creates an adversarial MCTS agent (standard alternating zero-sum search).
    pub fn new_adversarial(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self::new_with_mode(
            name,
            num_iterations,
            c_puct,
            model,
            MctsMode::Adversarial,
            verbose,
        )
    }

    /// Creates an MCTS agent with custom evaluation model and parameters (defaults to Adversarial mode).
    pub fn new_with_model(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self::new_adversarial(name, num_iterations, c_puct, model, verbose)
    }

    /// Creates a macro-action MCTS agent assuming the opponent follows a tactical heuristic.
    pub fn new_macro_tactical(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self::new_with_mode(
            name,
            num_iterations,
            c_puct,
            model,
            MctsMode::MacroTactical,
            verbose,
        )
    }

    /// Alias for [`new_macro_tactical`](Self::new_macro_tactical).
    pub fn new_tactical(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self::new_macro_tactical(name, num_iterations, c_puct, model, verbose)
    }

    /// Creates a macro-action MCTS agent assuming the opponent chooses uniform random moves.
    pub fn new_macro_random(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self::new_with_mode(
            name,
            num_iterations,
            c_puct,
            model,
            MctsMode::MacroRandom,
            verbose,
        )
    }

    /// Alias for [`new_macro_random`](Self::new_macro_random).
    pub fn new_random(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self::new_macro_random(name, num_iterations, c_puct, model, verbose)
    }

    /// Returns the agent display name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<const R: usize, const C: usize> MctsAgent<RolloutEvaluator<R, C>, R, C> {
    /// Creates an MCTS agent guided by random rollout simulations.
    pub fn new_rollout(
        name: impl Into<String>,
        num_iterations: usize,
        num_rollouts: usize,
        max_depth: usize,
        verbose: bool,
    ) -> Self {
        Self::new_adversarial(
            name,
            num_iterations,
            1.414,
            RolloutEvaluator::new(num_rollouts, max_depth),
            verbose,
        )
    }
}

impl<const R: usize, const C: usize> MctsAgent<UniformEvaluator, R, C> {
    /// Creates an MCTS agent with uniform priors and zero heuristic values.
    pub fn new_uniform(name: impl Into<String>, num_iterations: usize, verbose: bool) -> Self {
        Self::new_adversarial(name, num_iterations, 1.414, UniformEvaluator, verbose)
    }
}

/// Backwards-compatible alias for round-based MCTS agents.
pub type RoundMctsAgent<M, const R: usize = 6, const C: usize = 7> = MctsAgent<M, R, C>;

/// Backwards-compatible alias for macro-action MCTS agents.
pub type MacroMctsAgent<M, const R: usize = 6, const C: usize = 7> = MctsAgent<M, R, C>;

#[allow(clippy::too_many_arguments)]
fn run_mcts_search<D, M, const R: usize, const C: usize, StepDelta>(
    dynamics: &D,
    model: &M,
    state: &Connect4State<R, C>,
    legal: &[usize],
    c_puct: f32,
    num_iterations: usize,
    agent_name: &str,
    verbose: bool,
    root_agent: AgentId,
) -> usize
where
    StepDelta: PartialEq + Clone,
    D: mcts_traits::AgentDynamics<
            Action = usize,
            Reward = [f32; 2],
            StepDelta = StepDelta,
            State = Connect4State<R, C>,
        >,
    M: Model<Connect4State<R, C>>,
{
    let selection = MultiAgentPuctSelection::<2> { c_puct };
    let backup = VectorBackup::<2>::default();
    let stats = MultiAgentPuctStats::<2>::new();

    let node_cap = num_iterations + 16;
    let edge_cap = node_cap * C;
    let mut tree = TreeStore::with_capacity(node_cap, edge_cap, stats);
    let root = tree.insert_root(root_agent);

    let scheduler = SequentialScheduler;
    scheduler.search(
        &mut tree,
        dynamics,
        model,
        &selection,
        &backup,
        root,
        state,
        num_iterations,
    );

    // Collect candidate statistics
    let num_children = tree.num_children(root);
    let first_edge = tree.first_child_edge(root);
    let total_root_visits: u32 = tree
        .child_edges(root)
        .map(|e| tree.stats.visits[e.as_usize()])
        .sum();

    let mut candidates = Vec::with_capacity(num_children as usize);
    for i in 0..num_children {
        let edge = mcts_engine::tree_store::EdgeId(first_edge.0 + i);
        let edge_idx = edge.as_usize();
        let action = *tree.edge_action(edge);
        let visits = tree.stats.visits[edge_idx];
        let visit_fraction = if total_root_visits > 0 {
            visits as f32 / total_root_visits as f32
        } else {
            0.0
        };
        let prior = tree.stats.priors[edge_idx];
        let mean_value = tree.stats.mean_value[edge_idx];

        candidates.push(MoveCandidate {
            action,
            visits,
            visit_fraction,
            prior,
            mean_value,
        });
    }

    if verbose {
        println!("\n[MCTS Search Analysis: {agent_name}]");
        print!("{}", format_move_candidates(&candidates));
    }

    // Select action with maximum visit count
    candidates
        .iter()
        .max_by_key(|c| c.visits)
        .map(|c| c.action)
        .unwrap_or(legal[0])
}

impl<M, const R: usize, const C: usize> mcts_traits::Agent<Connect4State<R, C>, usize>
    for MctsAgent<M, R, C>
where
    M: Model<Connect4State<R, C>>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Connect4State<R, C>) -> usize {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("MctsAgent: no legal actions available in state");
        }
        if legal.len() == 1 {
            return legal[0];
        }

        let primary_agent = AgentId(state.current_player.index() as u32);
        match self.mode {
            MctsMode::Adversarial => {
                let dynamics = TurnBasedDynamics::new(Connect4World::<R, C>::new());
                run_mcts_search(
                    &dynamics,
                    &self.model,
                    state,
                    &legal,
                    self.c_puct,
                    self.num_iterations,
                    &self.name,
                    self.verbose,
                    primary_agent,
                )
            }
            MctsMode::MacroTactical => {
                let dynamics = crate::dynamics::MacroConnect4Dynamics::new(
                    TacticalOpponent,
                    state.current_player,
                );
                run_mcts_search(
                    &dynamics,
                    &self.model,
                    state,
                    &legal,
                    self.c_puct,
                    self.num_iterations,
                    &self.name,
                    self.verbose,
                    primary_agent,
                )
            }
            MctsMode::MacroRandom => {
                let dynamics = crate::dynamics::MacroConnect4Dynamics::new(
                    crate::dynamics::RandomOpponent::new(),
                    state.current_player,
                );
                run_mcts_search(
                    &dynamics,
                    &self.model,
                    state,
                    &legal,
                    self.c_puct,
                    self.num_iterations,
                    &self.name,
                    self.verbose,
                    primary_agent,
                )
            }
        }
    }
}

/// Specification for constructing a Connect 4 agent from CLI arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Connect4AgentSpec {
    /// Baseline agent playing uniform random legal moves.
    Random,
    /// Greedy tactical heuristic (immediate win, block 1-ply win, center preference).
    Tactical,
    /// Standard zero-sum adversarial MCTS.
    Mcts { iters: usize, rollouts: usize },
    /// Macro-action MCTS assuming opponent follows a tactical heuristic.
    MacroTactical { iters: usize, rollouts: usize },
    /// Macro-action MCTS assuming opponent chooses uniform random moves.
    MacroRandom { iters: usize, rollouts: usize },
}

impl Connect4AgentSpec {
    /// Parses an agent specification string (e.g. `mcts:200:3`, `macro-tactical:100`, `tactical`, `random`).
    pub fn parse(s: &str, default_iters: usize, default_rollouts: usize) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        let parse_iters = |idx: usize| -> Result<usize, String> {
            if parts.len() > idx {
                parts[idx]
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid iteration count in '{s}'"))
            } else {
                Ok(default_iters)
            }
        };

        let parse_rollouts = |idx: usize| -> Result<usize, String> {
            if parts.len() > idx {
                parts[idx]
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid rollouts count in '{s}'"))
            } else {
                Ok(default_rollouts)
            }
        };

        match parts[0].trim().to_lowercase().as_str() {
            "random" | "rand" => Ok(Self::Random),
            "tactical" | "heur" | "heuristic" => Ok(Self::Tactical),
            "mcts" | "mcts-sequential" | "sequential" | "adversarial" | "round-adversarial"
            | "round-adv" | "round" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::Mcts { iters, rollouts })
            }
            "macro-tactical"
            | "macro-mcts-tactical"
            | "macro-tact"
            | "macro"
            | "tactical-mcts"
            | "round-tactical"
            | "round-tact" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::MacroTactical { iters, rollouts })
            }
            "macro-random" | "macro-mcts-random" | "macro-rand" | "random-mcts"
            | "round-random" | "round-rand" => {
                let iters = parse_iters(1)?;
                let rollouts = parse_rollouts(2)?;
                Ok(Self::MacroRandom { iters, rollouts })
            }
            other => Err(format!(
                "Unknown agent type '{other}'. Supported: mcts[:iters[:rollouts]], macro-tactical[:iters[:rollouts]], macro-random[:iters[:rollouts]], tactical, random"
            )),
        }
    }

    /// Returns a human-readable display name summarizing type and hyperparameters.
    pub fn display_name(&self) -> String {
        match self {
            Self::Random => "Random".to_string(),
            Self::Tactical => "Tactical".to_string(),
            Self::Mcts { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("MCTS({iters},{eval})")
            }
            Self::MacroTactical { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Macro-Tact({iters},{eval})")
            }
            Self::MacroRandom { iters, rollouts } => {
                let eval = if *rollouts == 0 { "uniform" } else { "rollout" };
                format!("Macro-Rand({iters},{eval})")
            }
        }
    }

    /// Instantiates an agent trait object ready for game execution.
    pub fn instantiate(&self, name: &str, c_puct: f32, verbose: bool) -> BoxAgent<6, 7> {
        match self {
            Self::Random => Box::new(RandomAgent::new(name)),
            Self::Tactical => Box::new(TacticalAgent::new(name)),
            Self::Mcts { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_adversarial(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_adversarial(
                        name,
                        *iters,
                        c_puct,
                        RolloutEvaluator::new(*rollouts, 20),
                        verbose,
                    ))
                }
            }
            Self::MacroTactical { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_macro_tactical(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_macro_tactical(
                        name,
                        *iters,
                        c_puct,
                        RolloutEvaluator::new(*rollouts, 20),
                        verbose,
                    ))
                }
            }
            Self::MacroRandom { iters, rollouts } => {
                if *rollouts == 0 {
                    Box::new(MctsAgent::new_macro_random(
                        name,
                        *iters,
                        c_puct,
                        UniformEvaluator,
                        verbose,
                    ))
                } else {
                    Box::new(MctsAgent::new_macro_random(
                        name,
                        *iters,
                        c_puct,
                        RolloutEvaluator::new(*rollouts, 20),
                        verbose,
                    ))
                }
            }
        }
    }
}

/// Disambiguates duplicate agent names by appending sequential numeric suffixes.
pub fn generate_unique_names(specs: &[Connect4AgentSpec]) -> Vec<String> {
    let raw_names: Vec<String> = specs.iter().map(|s| s.display_name()).collect();
    mcts_engine::arena::disambiguate_names(&raw_names)
}
