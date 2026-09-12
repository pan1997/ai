//! # Blokus
//!
//! High-performance, zero-allocation Blokus board game engine, multi-agent MCTS agents,
//! and interactive terminal players.
//!
//! ## Variants Supported
//! - **Blokus Classic**: 4-player game on a $20 \times 20$ board starting from the 4 corners.
//! - **Blokus Duo**: 2-player game on a $14 \times 14$ board starting at $(4, 4)$ and $(9, 9)$.

pub mod agent;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod pieces;
pub mod render;
pub mod world;

#[cfg(test)]
mod tests;

pub use agent::{Agent, HeuristicAgent, HumanAgent, MctsAgent, RandomAgent};
pub use dynamics::compute_rank_rewards;
pub use evaluator::{
    AreaHeuristicEvaluator, BlokusEvaluator, HeuristicRolloutEvaluator, HeuristicUtilityEvaluator,
    RolloutEvaluator, UniformEvaluator,
};
pub use game::{BlokusAction, BlokusState, EMPTY, Player};
pub use pieces::{
    NUM_PIECES, PIECE_NAMES, PieceRegistry, PolyominoShape, TOTAL_SQUARES_PER_PLAYER, piece_size,
    registry,
};
pub use render::{
    MoveCandidate, format_move_candidates, render_board, render_inventory, render_scoreboard,
};
pub use world::{BlokusClassicWorld, BlokusDuoWorld, BlokusWorld};

/// Standard 4-player Blokus Classic state on a $20 \times 20$ board.
pub type BlokusClassicState = BlokusState<20, 4>;

/// Standard 2-player Blokus Duo state on a $14 \times 14$ board.
pub type BlokusDuoState = BlokusState<14, 2>;
