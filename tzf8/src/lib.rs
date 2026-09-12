//! High-performance 2048 game engine, Expectimax chance-node MCTS agents, and benchmark tournament arena.
//!
//! # Architecture
//!
//! - [`game`]: Fast $4 \times 4$ board representation, directional shifts, merges, and tile spawn events.
//! - [`world`]: Impartial referee and match driver implementing [`mcts_traits::World`].
//! - [`dynamics`]: Single-agent transition dynamics implementing [`mcts_traits::AgentDynamics`]
//!   with stochastic chance branching via [`game::TileSpawn`].
//! - [`evaluator`]: Leaf evaluation models ([`CornerHeuristicEvaluator`], [`RolloutEvaluator`], [`UniformEvaluator`]).
//! - [`agent`]: Agent traits and implementations ([`HumanAgent`], [`RandomAgent`], [`HeuristicAgent`], [`MctsAgent`]).
//! - [`render`]: ANSI terminal board rendering and search candidate move inspection.

pub mod agent;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod render;
pub mod world;

pub use agent::{BoxAgent, HeuristicAgent, HumanAgent, MctsAgent, RandomAgent, Tzf8AgentSpec};
pub use dynamics::Tzf8Dynamics;
pub use evaluator::{CornerHeuristicEvaluator, RolloutEvaluator, UniformEvaluator};
pub use game::{Direction, TileSpawn, Tzf8State};
pub use render::{MoveCandidate, format_move_candidates, format_tile, render_board};
pub use world::Tzf8World;

#[cfg(test)]
mod tests;
