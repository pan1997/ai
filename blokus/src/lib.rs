//! # Blokus
//!
//! High-performance, zero-allocation Blokus board game engine, multi-agent MCTS agents,
//! and interactive terminal players.
//!
//! ## Variants Supported
//! - **Blokus Classic**: 4-player game on a $20 \times 20$ board starting from the 4 corners.
//! - **Blokus Duo**: 2-player game on a $14 \times 14$ board starting at $(4, 4)$ and $(9, 9)$.

pub mod pieces;
pub mod game;
pub mod world;
pub mod dynamics;
pub mod evaluator;
pub mod render;
pub mod agent;

#[cfg(test)]
mod tests;

pub use pieces::{
    piece_size, registry, PolyominoShape, PieceRegistry, NUM_PIECES, PIECE_NAMES,
    TOTAL_SQUARES_PER_PLAYER,
};
pub use game::{BlokusAction, BlokusState, Player, EMPTY};
pub use dynamics::{compute_rank_rewards, BlokusDynamics};
pub use world::{BlokusClassicWorld, BlokusDuoWorld, BlokusWorld};
pub use evaluator::{
    AreaHeuristicEvaluator, HeuristicRolloutEvaluator, HeuristicUtilityEvaluator, RolloutEvaluator,
    UniformEvaluator,
};
pub use render::{
    format_move_candidates, render_board, render_inventory, render_scoreboard, MoveCandidate,
};
pub use agent::{Agent, HeuristicAgent, HumanAgent, MctsAgent, RandomAgent};

/// Standard 4-player Blokus Classic state on a $20 \times 20$ board.
pub type BlokusClassicState = BlokusState<20, 4>;

/// Standard 2-player Blokus Duo state on a $14 \times 14$ board.
pub type BlokusDuoState = BlokusState<14, 2>;

/// Standard 4-player Blokus Classic dynamics.
pub type BlokusClassicDynamics = BlokusDynamics<20, 4>;

/// Standard 2-player Blokus Duo dynamics.
pub type BlokusDuoDynamics = BlokusDynamics<14, 2>;
