//! Connect 4 Agent abstractions, Human CLI interaction, Random, and MCTS player implementations.

use crate::evaluator::{RolloutEvaluator, UniformEvaluator};
use crate::game::Connect4State;
use crate::render::{format_move_candidates, MoveCandidate};
use crate::world::Connect4World;
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentId, Model, TurnBasedDynamics};
use rand::seq::SliceRandom;
use std::io::{self, BufRead, Write};

/// General agent interface capable of selecting moves in a Connect 4 game.
pub trait Agent<const R: usize = 6, const C: usize = 7> {
    /// Returns the human-readable display name of the agent.
    fn name(&self) -> &str;

    /// Selects a legal column action $0 \le c < C$ given the current board state.
    fn select_action(&mut self, state: &Connect4State<R, C>) -> usize;
}

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

impl<const R: usize, const C: usize> Agent<R, C> for HumanAgent {
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

impl<const R: usize, const C: usize> Agent<R, C> for RandomAgent {
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

/// Monte Carlo Tree Search agent powered by `mcts-engine`.
pub struct MctsAgent<M, const R: usize = 6, const C: usize = 7> {
    name: String,
    pub num_iterations: usize,
    pub c_puct: f32,
    pub model: M,
    pub verbose: bool,
}

impl<M, const R: usize, const C: usize> MctsAgent<M, R, C> {
    /// Creates a new MCTS agent with custom evaluation model and parameters.
    pub fn new_with_model(
        name: impl Into<String>,
        num_iterations: usize,
        c_puct: f32,
        model: M,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct,
            model,
            verbose,
        }
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
        Self {
            name: name.into(),
            num_iterations,
            c_puct: 1.414,
            model: RolloutEvaluator::new(num_rollouts, max_depth),
            verbose,
        }
    }
}

impl<const R: usize, const C: usize> MctsAgent<UniformEvaluator, R, C> {
    /// Creates an MCTS agent with uniform priors and zero heuristic values.
    pub fn new_uniform(name: impl Into<String>, num_iterations: usize, verbose: bool) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct: 1.414,
            model: UniformEvaluator,
            verbose,
        }
    }
}

impl<M, const R: usize, const C: usize> Agent<R, C> for MctsAgent<M, R, C>
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

        let dynamics = TurnBasedDynamics::new(Connect4World::<R, C>::new());
        let selection = MultiAgentPuctSelection::<2> {
            c_puct: self.c_puct,
        };
        let backup = VectorBackup::<2>::default();
        let stats = MultiAgentPuctStats::<2>::new();

        let node_cap = self.num_iterations + 16;
        let edge_cap = node_cap * C;
        let mut tree = TreeStore::with_capacity(node_cap, edge_cap, stats);
        let root = tree.insert_root(AgentId(state.current_player.index() as u32));

        let scheduler = SequentialScheduler;
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

        if self.verbose {
            println!("\n[MCTS Search Analysis: {}]", self.name);
            print!("{}", format_move_candidates(&candidates));
        }

        // Select action with maximum visit count
        candidates
            .iter()
            .max_by_key(|c| c.visits)
            .map(|c| c.action)
            .unwrap_or(legal[0])
    }
}
