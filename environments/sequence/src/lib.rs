//! Dedicated Sequence game engine, MCTS agents, and interactive players.
//!
//! Sequence is a board-and-card strategy game for 2 to 6 players played across 2 or 3 teams.
//! Players take turns playing cards from their private hands to place tokens on matching spaces
//! of a $10 \times 10$ board, aiming to complete connected 5-token sequences horizontally,
//! vertically, or diagonally. Two-Eyed Jacks act as wildcards, and One-Eyed Jacks remove opponent
//! tokens.
//!
//! ## Key Modules
//! - [`board`]: Board card layout matrix, coordinate conversion, and localized $O(1)$ sequence detection.
//! - [`game`]: Core rules, actions, multi-player/team state transitions, and dead-card rules.
//! - [`world`]: Ground-truth referee implementing [`World`](mcts_traits::World) and [`TurnBasedWorld`](mcts_traits::TurnBasedWorld) with imperfect information observations.
//! - [`dynamics`]: Observation determinization and opponent-modeled macro dynamics.
//! - [`evaluator`]: Domain heuristic threat evaluators and rollout models.
//! - [`render`]: Terminal board rendering with ANSI team token colors.
//! - [`agent`]: Human CLI, Random, Heuristic, ISMCTS, and Opponent-Model MCTS agents.

pub mod agent;
pub mod board;
pub mod dynamics;
pub mod evaluator;
pub mod game;
pub mod render;
pub mod world;

#[cfg(test)]
mod tests;

pub use agent::{
    parse_agent, BoxAgent, HeuristicAgent, HumanAgent, IsMctsAgent, MctsAgent,
    OpponentModelMctsAgent, RandomAgent,
};
pub use board::{
    coord_to_index, create_double_deck, index_to_coord, is_corner, is_corner_index,
    is_one_eyed_jack, is_two_eyed_jack, BoardCell, Card, SequenceRecord, BOARD_CELLS, BOARD_DIM,
};
pub use dynamics::{
    determinize_state, RandomOpponentPolicy, SequenceRoundDynamics, SequenceTurnDynamics,
};
pub use evaluator::{SequenceHeuristicEvaluator, UniformEvaluator};
pub use game::{SequenceAction, SequenceConfig, SequenceState};
pub use render::{format_action, format_card, render_board, render_state};
pub use world::{
    Sequence2PWorld, Sequence3PWorld, Sequence4PWorld, Sequence6PWorld, SequenceObservation,
    SequenceWorld,
};

