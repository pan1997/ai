//! Agent implementations, Human CLI interaction, Heuristic, Random, and MCTS players for 2048.

use crate::dynamics::Tzf8Dynamics;
use crate::evaluator::{CornerHeuristicEvaluator, RolloutEvaluator, UniformEvaluator};
use crate::game::{Direction, TileSpawn, Tzf8State};
use crate::render::{MoveCandidate, format_move_candidates};
use mcts_engine::backup::SingleAgentBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{
    MultiAgentPuctStats, NormalizedPuctSelection, NormalizedUctSelection, UctSelection,
};
use mcts_engine::tree_store::{EdgeId, TreeStore};
use mcts_traits::{AgentDynamics, AgentId, Model};
use rand::seq::SliceRandom;
use std::io::{self, BufRead, Write};

pub use mcts_traits::Agent;

/// Dynamic boxed 2048 agent.
pub type BoxAgent = Box<dyn Agent<Tzf8State, Direction>>;

/// Interactive human player prompting for moves via standard input.
pub struct HumanAgent {
    name: String,
}

impl HumanAgent {
    /// Creates a new human player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for HumanAgent {
    fn default() -> Self {
        Self::new("Human")
    }
}

impl Agent<Tzf8State, Direction> for HumanAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Tzf8State) -> Direction {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut legal = Vec::new();
        state.legal_directions(&mut legal);

        loop {
            print!(
                "{} - enter direction [W/A/S/D or Up/Left/Down/Right] (legal: {:?}): ",
                self.name,
                legal.iter().map(|d| d.name()).collect::<Vec<_>>()
            );
            let _ = stdout.flush();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).is_err() {
                println!("Error reading input, please try again.");
                continue;
            }

            let trimmed = input.trim().to_lowercase();
            let chosen = match trimmed.as_str() {
                "w" | "up" | "u" => Some(Direction::Up),
                "a" | "left" | "l" => Some(Direction::Left),
                "s" | "down" => Some(Direction::Down),
                "d" | "right" | "r" => Some(Direction::Right),
                "q" | "quit" => {
                    println!("Quitting game.");
                    std::process::exit(0);
                }
                _ => {
                    println!("Unrecognized input '{trimmed}'. Use W/A/S/D or arrow names.");
                    None
                }
            };

            if let Some(dir) = chosen {
                if legal.contains(&dir) {
                    return dir;
                }
                println!(
                    "Direction {} is not a legal move (tiles cannot slide).",
                    dir.name()
                );
            }
        }
    }
}

/// Baseline agent selecting uniformly at random among legal directions.
pub struct RandomAgent {
    name: String,
}

impl RandomAgent {
    /// Creates a new random player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self::new("Random")
    }
}

impl Agent<Tzf8State, Direction> for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Tzf8State) -> Direction {
        let mut legal = Vec::new();
        state.legal_directions(&mut legal);
        let mut rng = rand::thread_rng();
        *legal
            .choose(&mut rng)
            .expect("RandomAgent: no legal moves available in state")
    }
}

/// Greedy 1-ply agent choosing the direction that maximizes the corner heuristic score.
pub struct HeuristicAgent {
    name: String,
    evaluator: CornerHeuristicEvaluator,
}

impl HeuristicAgent {
    /// Creates a new heuristic agent with default weights.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            evaluator: CornerHeuristicEvaluator::new(),
        }
    }
}

impl Default for HeuristicAgent {
    fn default() -> Self {
        Self::new("Heuristic")
    }
}

impl Agent<Tzf8State, Direction> for HeuristicAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Tzf8State) -> Direction {
        let mut legal = Vec::new();
        state.legal_directions(&mut legal);
        if legal.is_empty() {
            panic!("HeuristicAgent: no legal moves available in state");
        }

        let mut best_dir = legal[0];
        let mut best_score = f32::NEG_INFINITY;

        for &dir in &legal {
            let mut next = state.clone();
            let (changed, score_gain) = next.move_board(dir);
            if !changed {
                continue;
            }
            let heuristic_val = self.evaluator.evaluate_state(&next);
            let total = score_gain as f32 + heuristic_val;
            if total > best_score {
                best_score = total;
                best_dir = dir;
            }
        }

        best_dir
    }
}

/// Selection policy strategy for 2048 MCTS planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionMode {
    /// Classic unnormalized UCT selection policy.
    Uct,
    /// Min-Max normalized UCT selection policy (dynamically normalizes $Q \in [0, 1]$).
    #[default]
    NormalizedUct,
    /// Min-Max normalized PUCT selection policy (normalizes $Q \in [0, 1]$ with policy priors).
    NormalizedPuct,
}

/// Monte Carlo Tree Search agent with stochastic chance-node branching for 2048.
pub struct MctsAgent<M = CornerHeuristicEvaluator> {
    name: String,
    /// Number of MCTS simulation sweeps per move.
    pub num_iterations: usize,
    /// PUCT/UCT exploration constant.
    pub c_puct: f32,
    /// Future reward discount factor $\gamma \in [0, 1]$.
    pub gamma: f32,
    /// Leaf state evaluation model.
    pub model: M,
    /// Selection policy strategy (UCT, NormalizedUct, NormalizedPuct).
    pub selection_mode: SelectionMode,
    /// Whether to print candidate move visit statistics to stdout.
    pub verbose: bool,
}

impl<M> MctsAgent<M> {
    /// Creates a new MCTS agent with classic unnormalized UCT selection.
    pub fn new(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        gamma: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            gamma,
            model,
            selection_mode: SelectionMode::Uct,
            verbose,
        }
    }

    /// Creates a new MCTS agent with Min-Max normalized UCT selection.
    pub fn new_normalized(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        gamma: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            gamma,
            model,
            selection_mode: SelectionMode::NormalizedUct,
            verbose,
        }
    }

    /// Creates a new MCTS agent with Min-Max normalized PUCT selection.
    pub fn new_puct(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        gamma: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            gamma,
            model,
            selection_mode: SelectionMode::NormalizedPuct,
            verbose,
        }
    }

    /// Creates a new MCTS agent with custom selection mode.
    pub fn new_with_selection(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        gamma: f32,
        model: M,
        selection_mode: SelectionMode,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            gamma,
            model,
            selection_mode,
            verbose,
        }
    }
}

impl<M: Model<Tzf8State>> Agent<Tzf8State, Direction> for MctsAgent<M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &Tzf8State) -> Direction {
        let dynamics = Tzf8Dynamics::new();
        let mut legal = Vec::new();
        dynamics.actions(state, &mut legal);
        if legal.len() <= 1 {
            return legal[0];
        }

        let backup = SingleAgentBackup::new(self.gamma);
        let scheduler = SequentialScheduler;

        let stats = MultiAgentPuctStats::<1>::new();
        let mut tree: TreeStore<Direction, [f32; 1], _, Option<TileSpawn>> =
            TreeStore::with_capacity(self.num_iterations * 2, self.num_iterations * 2, stats);

        let root = tree.insert_root(AgentId(0));
        match self.selection_mode {
            SelectionMode::Uct => {
                let selection = UctSelection::<1> { c_uct: self.c_puct };
                scheduler.search(
                    &mut tree,
                    &dynamics,
                    &self.model,
                    &selection,
                    &backup,
                    root,
                    state,
                    self.num_iterations,
                );
            }
            SelectionMode::NormalizedUct => {
                let selection = NormalizedUctSelection::<1> { c_uct: self.c_puct };
                scheduler.search(
                    &mut tree,
                    &dynamics,
                    &self.model,
                    &selection,
                    &backup,
                    root,
                    state,
                    self.num_iterations,
                );
            }
            SelectionMode::NormalizedPuct => {
                let selection = NormalizedPuctSelection::<1> {
                    c_puct: self.c_puct,
                };
                scheduler.search(
                    &mut tree,
                    &dynamics,
                    &self.model,
                    &selection,
                    &backup,
                    root,
                    state,
                    self.num_iterations,
                );
            }
        }

        let num_children = tree.num_children(root);
        let first_edge = tree.first_child_edge(root);

        let mut candidates = Vec::with_capacity(num_children as usize);
        let mut best_edge = EdgeId::INVALID;
        let mut best_visits = 0;
        let mut best_q = f32::NEG_INFINITY;

        for i in 0..num_children {
            let edge = EdgeId(first_edge.0 + i);
            let edge_idx = edge.as_usize();
            let dir = *tree.edge_action(edge);
            let visits = tree.stats.visits[edge_idx];
            let mean_q = tree.stats.mean_value[edge_idx][0];
            let prior = tree.stats.priors[edge_idx];

            candidates.push(MoveCandidate {
                direction: dir,
                visits,
                mean_q,
                prior,
            });

            if visits > best_visits || (visits == best_visits && mean_q > best_q) {
                best_visits = visits;
                best_q = mean_q;
                best_edge = edge;
            }
        }

        if self.verbose {
            print!("{}", format_move_candidates(&candidates));
        }

        assert!(best_edge.is_valid(), "MCTS failed to select a valid edge");
        *tree.edge_action(best_edge)
    }
}

/// Agent specification string parser for CLI configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum Tzf8AgentSpec {
    /// Interactive human terminal player.
    Human,
    /// Uniform random choice.
    Random,
    /// 1-ply greedy corner heuristic.
    Heuristic,
    /// MCTS with corner heuristic model: `mcts[:iters[:c_puct]]`.
    Mcts { iters: usize, c_puct: f32 },
    /// MCTS with Min-Max normalized UCT: `mcts-norm[:iters[:c_puct]]`.
    MctsNorm { iters: usize, c_puct: f32 },
    /// MCTS with Min-Max normalized PUCT: `mcts-puct[:iters[:c_puct]]`.
    MctsPuct { iters: usize, c_puct: f32 },
    /// MCTS with random rollout evaluation: `mcts-rollout[:iters[:rollouts]]`.
    MctsRollout { iters: usize, rollouts: usize },
    /// MCTS with uniform zero model: `mcts-uniform[:iters[:c_puct]]`.
    MctsUniform { iters: usize, c_puct: f32 },
}

impl Tzf8AgentSpec {
    /// Parses an agent specification string (e.g. `heuristic`, `mcts:500`, `random`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts[0] {
            "human" => Ok(Self::Human),
            "random" => Ok(Self::Random),
            "heuristic" => Ok(Self::Heuristic),
            "mcts" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid MCTS iters: {e}"))?
                } else {
                    200
                };
                let c_puct = if parts.len() > 2 {
                    parts[2]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid MCTS c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::Mcts { iters, c_puct })
            }
            "mcts-norm" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid MCTS iters: {e}"))?
                } else {
                    200
                };
                let c_puct = if parts.len() > 2 {
                    parts[2]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid MCTS c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::MctsNorm { iters, c_puct })
            }
            "mcts-puct" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid MCTS iters: {e}"))?
                } else {
                    200
                };
                let c_puct = if parts.len() > 2 {
                    parts[2]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid MCTS c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::MctsPuct { iters, c_puct })
            }
            "mcts-rollout" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid MCTS iters: {e}"))?
                } else {
                    100
                };
                let rollouts = if parts.len() > 2 {
                    parts[2]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid MCTS rollouts: {e}"))?
                } else {
                    5
                };
                Ok(Self::MctsRollout { iters, rollouts })
            }
            "mcts-uniform" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid MCTS iters: {e}"))?
                } else {
                    200
                };
                let c_puct = if parts.len() > 2 {
                    parts[2]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid MCTS c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::MctsUniform { iters, c_puct })
            }
            other => Err(format!(
                "Unknown agent type '{other}'. Valid types: human, random, heuristic, mcts, mcts-norm, mcts-puct, mcts-rollout, mcts-uniform"
            )),
        }
    }

    /// Instantiates a dynamic boxed agent according to this specification.
    pub fn build_agent(&self) -> BoxAgent {
        match self {
            Self::Human => Box::new(HumanAgent::default()),
            Self::Random => Box::new(RandomAgent::default()),
            Self::Heuristic => Box::new(HeuristicAgent::default()),
            Self::Mcts { iters, c_puct } => Box::new(MctsAgent::new(
                format!("MCTS-Heuristic({iters})"),
                *iters,
                *c_puct,
                1.0,
                CornerHeuristicEvaluator::new(),
                false,
            )),
            Self::MctsNorm { iters, c_puct } => Box::new(MctsAgent::new_normalized(
                format!("MCTS-Norm({iters})"),
                *iters,
                *c_puct,
                1.0,
                CornerHeuristicEvaluator::new(),
                false,
            )),
            Self::MctsPuct { iters, c_puct } => Box::new(MctsAgent::new_puct(
                format!("MCTS-PUCT({iters})"),
                *iters,
                *c_puct,
                1.0,
                CornerHeuristicEvaluator::new(),
                false,
            )),
            Self::MctsRollout { iters, rollouts } => Box::new(MctsAgent::new(
                format!("MCTS-Rollout({iters},r={rollouts})"),
                *iters,
                1.414,
                1.0,
                RolloutEvaluator::new(*rollouts, 25),
                false,
            )),
            Self::MctsUniform { iters, c_puct } => Box::new(MctsAgent::new(
                format!("MCTS-Uniform({iters})"),
                *iters,
                *c_puct,
                1.0,
                UniformEvaluator,
                false,
            )),
        }
    }
}
