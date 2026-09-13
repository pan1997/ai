//! Autonomous and interactive agent implementations for Hex.

use crate::evaluator::{
    HexEvaluator, RolloutEvaluator, ShortestPathHeuristicEvaluator, UniformEvaluator,
};
use crate::game::{HexPlayer, HexState};
use crate::render::{MoveCandidate, format_move_candidates};
use crate::world::HexWorld;
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
pub use mcts_traits::Agent;
use mcts_traits::{AgentId, TurnBasedDynamics};
use rand::Rng;
use std::io::{self, BufRead, Write};

/// Trait object shorthand for boxed Hex agents.
pub type BoxAgent<const N: usize = 11> = Box<dyn Agent<HexState<N>, usize>>;

/// Interactive human player prompting move entries from standard input.
#[derive(Debug, Clone, Default)]
pub struct HumanAgent<const N: usize = 11> {
    name: String,
}

impl<const N: usize> HumanAgent<N> {
    /// Constructs a human player with default name "Human".
    pub fn new() -> Self {
        Self {
            name: "Human".to_string(),
        }
    }

    /// Constructs a human player with a custom display name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl<const N: usize> Agent<HexState<N>, usize> for HumanAgent<N> {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &HexState<N>) -> usize {
        let player = state.current_player;
        let color = player.color_code();
        let reset = "\x1b[0m";

        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        let swap_hint = if state.pie_rule && state.move_count == 1 && player == HexPlayer::White {
            ", 'swap'"
        } else {
            ""
        };

        loop {
            print!(
                "{color}[{} ({})]{reset} Enter move (e.g. F6{swap_hint}, 5 5, or index): ",
                self.name,
                player.char()
            );
            io::stdout().flush().unwrap();

            let line = match lines.next() {
                Some(Ok(l)) => l,
                _ => {
                    println!("\nEOF received. Selecting first legal move.");
                    let mut legal = Vec::new();
                    state.legal_actions(&mut legal);
                    return legal[0];
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(action) = HexState::<N>::str_to_coord(trimmed) {
                if action == HexState::<N>::SWAP_ACTION {
                    if state.pie_rule && state.move_count == 1 && player == HexPlayer::White {
                        return action;
                    } else {
                        println!("Pie Rule (swap) is not legal at this turn!");
                        continue;
                    }
                }
                if action < N * N && state.board[action].is_none() {
                    return action;
                } else {
                    println!(
                        "Cell {} ({}) is already occupied! Choose an empty cell.",
                        HexState::<N>::coord_to_str(action),
                        action
                    );
                }
            } else {
                println!(
                    "Invalid move syntax '{}'. Enter coordinate like 'F6'{swap_hint}, row & col like '5 5', or index.",
                    trimmed
                );
            }
        }
    }
}

/// Uniformly random player choosing from available unoccupied cells.
#[derive(Debug, Clone, Default)]
pub struct RandomAgent {
    name: String,
}

impl RandomAgent {
    /// Constructs a random agent with default name "Random".
    pub fn new() -> Self {
        Self {
            name: "Random".to_string(),
        }
    }

    /// Constructs a random agent with a custom display name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl<const N: usize> Agent<HexState<N>, usize> for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &HexState<N>) -> usize {
        let mut legal = Vec::with_capacity(N * N);
        state.legal_actions(&mut legal);
        assert!(!legal.is_empty(), "RandomAgent: no legal moves available");
        let mut rng = rand::thread_rng();
        legal[rng.gen_range(0..legal.len())]
    }
}

/// Fast 1-ply greedy lookahead agent selecting moves that optimize shortest-path boundary distance.
#[derive(Debug, Clone, Default)]
pub struct HeuristicAgent<const N: usize = 11> {
    name: String,
}

impl<const N: usize> HeuristicAgent<N> {
    /// Constructs a heuristic agent with default name "Heuristic".
    pub fn new() -> Self {
        Self {
            name: "Heuristic".to_string(),
        }
    }

    /// Constructs a heuristic agent with a custom display name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl<const N: usize> Agent<HexState<N>, usize> for HeuristicAgent<N> {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &HexState<N>) -> usize {
        let mut legal = Vec::with_capacity(N * N);
        state.legal_actions(&mut legal);
        assert!(
            !legal.is_empty(),
            "HeuristicAgent: no legal moves available"
        );

        let player = state.current_player;
        let mut best_action = legal[0];
        let mut best_score = match player {
            HexPlayer::Black => f32::NEG_INFINITY,
            HexPlayer::White => f32::INFINITY,
        };

        for &action in &legal {
            let mut next_state = state.clone();
            let won = next_state.play_move(action);
            if won {
                // Immediate winning move! Take it without hesitation.
                return action;
            }

            let score = ShortestPathHeuristicEvaluator::<N>::evaluate_state(&next_state);
            match player {
                HexPlayer::Black => {
                    if score > best_score {
                        best_score = score;
                        best_action = action;
                    }
                }
                HexPlayer::White => {
                    if score < best_score {
                        best_score = score;
                        best_action = action;
                    }
                }
            }
        }

        best_action
    }
}

/// High-performance Monte Carlo Tree Search agent for Hex.
pub struct MctsAgent<const N: usize = 11> {
    name: String,
    /// Number of MCTS simulation sweeps per move.
    pub num_iterations: usize,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Leaf evaluation strategy.
    pub evaluator: HexEvaluator<N>,
    /// Whether to print search candidate tables before committing to each move.
    pub verbose: bool,
}

impl<const N: usize> MctsAgent<N> {
    /// Creates a new MCTS agent with custom parameters and evaluator.
    pub fn new(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        evaluator: HexEvaluator<N>,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            evaluator,
            verbose,
        }
    }

    /// Creates an MCTS agent guided by random rollout simulations.
    pub fn new_rollout(
        name: impl Into<String>,
        num_iterations: usize,
        num_rollouts: usize,
        verbose: bool,
    ) -> Self {
        Self::new(
            name,
            num_iterations,
            1.414,
            HexEvaluator::Rollout(RolloutEvaluator::new(num_rollouts)),
            verbose,
        )
    }

    /// Creates an MCTS agent guided by the 0-1 BFS shortest path heuristic.
    pub fn new_heuristic(name: impl Into<String>, num_iterations: usize, verbose: bool) -> Self {
        Self::new(
            name,
            num_iterations,
            1.414,
            HexEvaluator::Heuristic(ShortestPathHeuristicEvaluator::new()),
            verbose,
        )
    }

    /// Creates an MCTS agent with uniform priors and zero heuristic values.
    pub fn new_uniform(name: impl Into<String>, num_iterations: usize, verbose: bool) -> Self {
        Self::new(
            name,
            num_iterations,
            1.414,
            HexEvaluator::Uniform(UniformEvaluator::new()),
            verbose,
        )
    }
}

impl<const N: usize> Agent<HexState<N>, usize> for MctsAgent<N> {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &HexState<N>) -> usize {
        let mut legal = Vec::with_capacity(N * N);
        state.legal_actions(&mut legal);
        assert!(!legal.is_empty(), "MctsAgent: no legal moves available");

        if legal.len() == 1 {
            return legal[0];
        }

        let world = HexWorld::<N>::with_pie_rule(state.pie_rule);
        let dynamics = TurnBasedDynamics::new(world);
        let active_player = state.current_player.index();

        let stats = MultiAgentPuctStats::<2>::new();
        let mut tree = TreeStore::with_capacity(
            (self.num_iterations + 10).min(50_000),
            ((self.num_iterations + 10) * legal.len()).min(500_000),
            stats,
        );

        let root = tree.insert_root(AgentId(active_player as u32));
        let selection = MultiAgentPuctSelection::<2> {
            c_puct: self.c_puct,
        };
        let backup = VectorBackup::<2>::default();
        let scheduler = SequentialScheduler;

        scheduler.search(
            &mut tree,
            &dynamics,
            &self.evaluator,
            &selection,
            &backup,
            root,
            state,
            self.num_iterations,
        );

        let mut candidates = Vec::with_capacity(legal.len());
        for edge in tree.child_edges(root) {
            let action = *tree.edge_action(edge);
            let edge_idx = edge.as_usize();
            let visits = tree.stats.visits[edge_idx];
            let q_value = tree.stats.mean_value[edge_idx][active_player];
            let prior = tree.stats.priors[edge_idx];

            candidates.push(MoveCandidate {
                action,
                coord: HexState::<N>::coord_to_str(action),
                visits,
                q_value,
                prior,
            });
        }

        candidates.sort_by_key(|a| std::cmp::Reverse(a.visits));

        if self.verbose {
            println!(
                "\n>>> {} ({}) Move Inspection ({} iters):",
                self.name,
                state.current_player.name(),
                self.num_iterations
            );
            print!("{}", format_move_candidates(&candidates, 10));
        }

        candidates.first().map(|c| c.action).unwrap_or(legal[0])
    }
}

/// Specifications for parsing agent configurations from CLI strings.
#[derive(Debug, Clone, PartialEq)]
pub enum HexAgentSpec {
    /// MCTS agent using random rollouts: `mcts[:iters[:rollouts]]`.
    MctsRollout {
        /// Number of MCTS simulation sweeps per move decision.
        iters: usize,
        /// Number of random simulation playouts per leaf node.
        rollouts: usize,
        /// Exploration constant scaling the prior policy influence.
        c_puct: f32,
    },
    /// MCTS agent using the shortest path heuristic: `mcts-h[:iters]`.
    MctsHeuristic {
        /// Number of MCTS simulation sweeps per move decision.
        iters: usize,
        /// Exploration constant scaling the prior policy influence.
        c_puct: f32,
    },
    /// MCTS agent using uniform priors: `mcts-u[:iters]`.
    MctsUniform {
        /// Number of MCTS simulation sweeps per move decision.
        iters: usize,
        /// Exploration constant scaling the prior policy influence.
        c_puct: f32,
    },
    /// 1-ply greedy lookahead heuristic agent: `heuristic`.
    Heuristic,
    /// Uniform random player: `random`.
    Random,
    /// Interactive human terminal player: `human`.
    Human,
}

impl HexAgentSpec {
    /// Parses an agent specification from a string slice (e.g. `"mcts:1000:2"`, `"heuristic"`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        let kind = parts[0].to_lowercase();

        match kind.as_str() {
            "mcts" | "rollout" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid iterations: {e}"))?
                } else {
                    500
                };
                let rollouts = if parts.len() > 2 {
                    parts[2]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid rollouts: {e}"))?
                } else {
                    2
                };
                let c_puct = if parts.len() > 3 {
                    parts[3]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::MctsRollout {
                    iters,
                    rollouts,
                    c_puct,
                })
            }
            "mcts-h" | "mcts-heuristic" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid iterations: {e}"))?
                } else {
                    500
                };
                let c_puct = if parts.len() > 2 {
                    parts[2]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::MctsHeuristic { iters, c_puct })
            }
            "mcts-u" | "mcts-uniform" => {
                let iters = if parts.len() > 1 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid iterations: {e}"))?
                } else {
                    500
                };
                let c_puct = if parts.len() > 2 {
                    parts[2]
                        .parse::<f32>()
                        .map_err(|e| format!("invalid c_puct: {e}"))?
                } else {
                    1.414
                };
                Ok(Self::MctsUniform { iters, c_puct })
            }
            "heuristic" | "h" => Ok(Self::Heuristic),
            "random" | "rand" | "r" => Ok(Self::Random),
            "human" => Ok(Self::Human),
            _ => Err(format!(
                "unknown agent spec '{s}'. Supported: mcts, mcts-h, mcts-u, heuristic, random, human"
            )),
        }
    }

    /// Instantiates the specified agent boxed for an $N \times N$ board.
    pub fn instantiate<const N: usize>(&self, name: String, verbose: bool) -> BoxAgent<N> {
        match self {
            Self::MctsRollout {
                iters,
                rollouts,
                c_puct,
            } => Box::new(MctsAgent::new(
                name,
                *iters,
                *c_puct,
                HexEvaluator::Rollout(RolloutEvaluator::new(*rollouts)),
                verbose,
            )),
            Self::MctsHeuristic { iters, c_puct } => Box::new(MctsAgent::new(
                name,
                *iters,
                *c_puct,
                HexEvaluator::Heuristic(ShortestPathHeuristicEvaluator::new()),
                verbose,
            )),
            Self::MctsUniform { iters, c_puct } => Box::new(MctsAgent::new(
                name,
                *iters,
                *c_puct,
                HexEvaluator::Uniform(UniformEvaluator::new()),
                verbose,
            )),
            Self::Heuristic => Box::new(HeuristicAgent::with_name(name)),
            Self::Random => Box::new(RandomAgent::with_name(name)),
            Self::Human => Box::new(HumanAgent::with_name(name)),
        }
    }

    /// Returns a standardized default name for the spec.
    pub fn default_name(&self) -> String {
        match self {
            Self::MctsRollout {
                iters, rollouts, ..
            } => format!("MCTS-Rollout({iters},r={rollouts})"),
            Self::MctsHeuristic { iters, .. } => format!("MCTS-Heuristic({iters})"),
            Self::MctsUniform { iters, .. } => format!("MCTS-Uniform({iters})"),
            Self::Heuristic => "Heuristic".to_string(),
            Self::Random => "Random".to_string(),
            Self::Human => "Human".to_string(),
        }
    }
}
