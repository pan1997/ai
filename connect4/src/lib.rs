//! # Connect 4 Game Engine & MCTS Toolkit
//!
//! High-performance, zero-allocation Connect 4 implementation supporting parametric
//! board dimensions $R \times C$, zero-copy `TreeStore` MCTS planning, and interactive
//! terminal gameplay.
//!
//! ## Modules
//! - [`game`]: Core board state, piece dropping, and bitboard/grid win checking.
//! - [`dynamics`]: Planning transition steps and two-player referee arbitration.
//! - [`evaluator`]: Prior policy and state value estimation models (rollouts, uniform).
//! - [`render`]: Terminal rendering with ANSI colors and MCTS search candidate tables.
//! - [`agent`]: Unified player interface with human, random, and MCTS implementations.

pub mod agent;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod render;

#[cfg(test)]
mod tests;

pub use agent::{Agent, HumanAgent, MctsAgent, RandomAgent};
pub use dynamics::Connect4Dynamics;
pub use evaluator::{RolloutEvaluator, UniformEvaluator};
pub use game::{Connect4State, Player};
pub use render::{format_move_candidates, render_board, render_board_styled, MoveCandidate};

