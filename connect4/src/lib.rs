//! # Connect 4 Game Engine & MCTS Toolkit
//!
//! High-performance, zero-allocation Connect 4 implementation supporting parametric
//! board dimensions $R \times C$, zero-copy `TreeStore` MCTS planning, and interactive
//! terminal gameplay.
//!
//! ## Modules
//! - [`game`]: Core board state, piece dropping, and bitboard/grid win checking.
//! - [`world`]: Impartial ground-truth referee and tournament match arbitration.
//! - [`dynamics`]: Planning transition steps (standard alternating and macro-action).
//! - [`evaluator`]: Prior policy and state value estimation models (rollouts, uniform).
//! - [`render`]: Terminal rendering with ANSI colors and MCTS search candidate tables.
//! - [`agent`]: Unified player interface with human, random, and MCTS implementations.

pub mod agent;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod render;
pub mod world;

#[cfg(test)]
mod tests;

pub use agent::{
    Agent, HumanAgent, MacroMctsAgent, MctsAgent, RandomAgent, RoundMctsAgent, TacticalAgent,
};
pub use dynamics::{MacroConnect4Dynamics, OpponentPolicy, RandomOpponent, TacticalOpponent};
pub use evaluator::{RolloutEvaluator, UniformEvaluator};
pub use game::{Connect4State, Player};
pub use render::{MoveCandidate, format_move_candidates, render_board, render_board_styled};
pub use world::Connect4World;
