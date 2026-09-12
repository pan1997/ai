//! High-performance Hex game engine, Disjoint Set Union (DSU) win tracking, MCTS agents, and tournament arena.
//!
//! # Architecture
//!
//! - [`game`]: Rhombus board representations, axial neighbors, DSU connectivity tracking, and algebraic notation parsing.
//! - [`world`]: Impartial 2-player match referee implementing [`mcts_traits::World`] and [`mcts_traits::TurnBasedWorld`].
//! - [`dynamics`]: Hypothetical single-agent planning transition dynamics via [`HexDynamics`].
//! - [`evaluator`]: Leaf evaluation models ([`ShortestPathHeuristicEvaluator`], [`RolloutEvaluator`], [`UniformEvaluator`]).
//! - [`agent`]: Agent traits and implementations ([`HumanAgent`], [`RandomAgent`], [`HeuristicAgent`], [`MctsAgent`]).
//! - [`render`]: ANSI terminal board rendering with rhombus layout and MCTS move candidate inspection.

pub mod agent;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod render;
pub mod world;

#[cfg(test)]
mod tests;

pub use agent::{
    Agent, BoxAgent, HeuristicAgent, HexAgentSpec, HumanAgent, MctsAgent, RandomAgent,
};
pub use dynamics::{HexDynamics, new_dynamics};
pub use evaluator::{
    HexEvaluator, RolloutEvaluator, ShortestPathHeuristicEvaluator, UniformEvaluator,
};
pub use game::{HexPlayer, HexState};
pub use render::{MoveCandidate, format_move_candidates, render_board, render_board_styled};
pub use world::HexWorld;
