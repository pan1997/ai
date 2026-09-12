//! Blokus Agent abstractions, Human CLI interaction, Random, Heuristic, and MCTS player implementations.

use crate::evaluator::{
    AreaHeuristicEvaluator, HeuristicRolloutEvaluator, HeuristicUtilityEvaluator, RolloutEvaluator,
    UniformEvaluator,
};
use crate::game::{BlokusAction, BlokusState, Player};
use crate::pieces::piece_size;
use crate::render::{MoveCandidate, format_move_candidates};
use crate::world::BlokusWorld;
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentId, Model, TurnBasedDynamics};
use rand::seq::SliceRandom;
use std::io::{self, BufRead, Write};

/// General agent interface capable of selecting actions in a Blokus match.
pub trait Agent<const B: usize = 20, const P: usize = 4> {
    /// Returns the human-readable display name of the agent.
    fn name(&self) -> &str;

    /// Selects a legal action given the current game state.
    fn select_action(&mut self, state: &BlokusState<B, P>) -> BlokusAction;
}

/// Interactive human player selecting moves via CLI prompts.
pub struct HumanAgent {
    name: String,
}

impl HumanAgent {
    /// Creates a new human player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl<const B: usize, const P: usize> Agent<B, P> for HumanAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &BlokusState<B, P>) -> BlokusAction {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);

        if legal.is_empty() || (legal.len() == 1 && legal[0] == BlokusAction::Pass) {
            println!("No legal moves available for {}. Passing turn.", self.name);
            return BlokusAction::Pass;
        }

        println!(
            "\n=== {}'s Turn ({}) ===",
            self.name,
            Player::from_index(state.current_player as usize)
        );
        println!(
            "Available legal moves ({} total). First 15 shown:",
            legal.len()
        );
        for (i, action) in legal.iter().take(15).enumerate() {
            println!("  [{:2}] {}", i, action);
        }
        if legal.len() > 15 {
            println!(
                "  ... and {} more options. (Enter 'list' to see all)",
                legal.len() - 15
            );
        }

        loop {
            print!(
                "Choose move index [0..{}] or 'list' or 'pass': ",
                legal.len() - 1
            );
            let _ = stdout.flush();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).is_err() {
                println!("Error reading input, please try again.");
                continue;
            }

            let trimmed = input.trim();
            if trimmed.eq_ignore_ascii_case("list") {
                for (i, action) in legal.iter().enumerate() {
                    println!("  [{:3}] {}", i, action);
                }
                continue;
            }

            if trimmed.eq_ignore_ascii_case("pass") {
                return BlokusAction::Pass;
            }

            match trimmed.parse::<usize>() {
                Ok(idx) if idx < legal.len() => {
                    return legal[idx];
                }
                _ => {
                    println!(
                        "Invalid input '{trimmed}'. Please enter a valid index 0..{}",
                        legal.len() - 1
                    );
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
}

impl<const B: usize, const P: usize> Agent<B, P> for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &BlokusState<B, P>) -> BlokusAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        let mut rng = rand::thread_rng();
        *legal.choose(&mut rng).unwrap_or(&BlokusAction::Pass)
    }
}

/// Greedy heuristic agent prioritizing placing larger polyominoes and maximizing new corners.
pub struct HeuristicAgent {
    name: String,
}

impl HeuristicAgent {
    /// Creates a new heuristic player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl<const B: usize, const P: usize> Agent<B, P> for HeuristicAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &BlokusState<B, P>) -> BlokusAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            return BlokusAction::Pass;
        }

        let mut best_score = i32::MIN;
        let mut best_action = legal[0];
        let p = state.current_player as usize;
        let mut corner_buf = Vec::new();

        for &action in &legal {
            let score = match action {
                BlokusAction::Pass => -1000,
                BlokusAction::Place { piece_id, .. } => {
                    let mut next = state.clone();
                    let _ = next.apply_action(&action);
                    next.find_valid_corners(p, &mut corner_buf);
                    let corners = corner_buf.len() as i32;
                    let size = piece_size(piece_id) as i32;
                    // Score = 10 * piece size + 2 * corners created
                    10 * size + 2 * corners
                }
            };

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }
}

/// High-performance multi-agent Monte Carlo Tree Search player.
pub struct MctsAgent<M, const B: usize = 20, const P: usize = 4> {
    name: String,
    /// Number of MCTS simulation iterations per move.
    pub num_iterations: usize,
    /// PUCT exploration constant $c_{\text{puct}}$.
    pub c_puct: f32,
    /// Evaluation model.
    pub model: M,
    /// Whether to print candidate move statistics to standard output.
    pub verbose: bool,
}

impl<M, const B: usize, const P: usize> MctsAgent<M, B, P> {
    /// Returns the player name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<const B: usize, const P: usize> MctsAgent<AreaHeuristicEvaluator, B, P> {
    /// Creates an MCTS agent guided by area heuristic evaluation.
    pub fn new_heuristic(name: impl Into<String>, num_iterations: usize, verbose: bool) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct: 1.414,
            model: AreaHeuristicEvaluator,
            verbose,
        }
    }
}

impl<const B: usize, const P: usize> MctsAgent<UniformEvaluator, B, P> {
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

impl<const B: usize, const P: usize> MctsAgent<RolloutEvaluator<B, P>, B, P> {
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

impl<const B: usize, const P: usize> MctsAgent<HeuristicUtilityEvaluator, B, P> {
    /// Creates an MCTS agent with heuristic utility priors and static territory values.
    pub fn new_heuristic_utility(
        name: impl Into<String>,
        num_iterations: usize,
        verbose: bool,
    ) -> Self {
        Self {
            name: name.into(),
            num_iterations,
            c_puct: 1.414,
            model: HeuristicUtilityEvaluator::default(),
            verbose,
        }
    }
}

impl<const B: usize, const P: usize> MctsAgent<HeuristicRolloutEvaluator<B, P>, B, P> {
    /// Creates an MCTS agent guided by informed heuristic simulation rollouts.
    pub fn new_heuristic_rollout(
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
            model: HeuristicRolloutEvaluator::new(num_rollouts, max_depth, 0.15),
            verbose,
        }
    }
}

impl<M, const B: usize, const P: usize> Agent<B, P> for MctsAgent<M, B, P>
where
    M: Model<BlokusState<B, P>>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &BlokusState<B, P>) -> BlokusAction {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            return BlokusAction::Pass;
        }
        if legal.len() == 1 {
            return legal[0];
        }

        let dynamics = TurnBasedDynamics::new(BlokusWorld::<B, P>::new());
        let selection = MultiAgentPuctSelection::<P> {
            c_puct: self.c_puct,
        };
        let backup = VectorBackup::<P>::default();
        let stats = MultiAgentPuctStats::<P>::new();

        let node_cap = self.num_iterations + 16;
        let edge_cap = node_cap * 64;
        let mut tree = TreeStore::with_capacity(node_cap, edge_cap, stats);
        let root = tree.insert_root(AgentId(state.current_player as u32));

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

        let num_children = tree.num_children(root);
        let first_edge = tree.first_child_edge(root);

        let mut candidates = Vec::with_capacity(num_children as usize);
        for i in 0..num_children {
            let edge = mcts_engine::tree_store::EdgeId(first_edge.0 + i);
            let edge_idx = edge.as_usize();
            let action = *tree.edge_action(edge);
            let visits = tree.stats.visits[edge_idx];
            let prior = tree.stats.priors[edge_idx];
            let mean_values = tree.stats.mean_value[edge_idx].to_vec();

            candidates.push(MoveCandidate {
                action,
                visits,
                prior,
                mean_values,
            });
        }

        candidates.sort_by_key(|c| std::cmp::Reverse(c.visits));

        if self.verbose {
            println!("\n[MCTS Search Analysis: {}]", self.name);
            print!("{}", format_move_candidates(&candidates, 10));
        }

        candidates.first().map(|c| c.action).unwrap_or(legal[0])
    }
}
