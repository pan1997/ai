//! # TicTacToe Environment
//!
//! Canonical 2-player TicTacToe game engine, MCTS planning agents, and CLI binaries.

pub mod agent;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod render;
pub mod world;

#[cfg(test)]
mod tests;

pub use agent::{BoxAgent, HumanAgent, MctsAgent, RandomAgent, TacticalAgent, parse_agent_spec};
pub use dynamics::TicTacToeDynamics;
pub use evaluator::{RolloutEvaluator, UniformEvaluator};
pub use game::{Player, TicTacToeState, WIN_LINES};
pub use render::{render_board, render_board_styled};
pub use world::TicTacToeWorld;
