//! # MCTS Utilities
//!
//! Visualization, Graphviz DOT generation, and graphical rendering utilities for Monte Carlo Tree Search
//! (MCTS) search trees built with [`mcts_engine::tree_store::TreeStore`].
//!
//! ## Overview
//!
//! `mcts-utils` provides high-performance graph generation and rendering tools for inspecting
//! search tree structure, visit distributions, Q-value estimates, policy priors, and principal variations.
//!
//! ## Key Capabilities
//!
//! - **Graphviz AST & DOT Generation**: [`to_dot`] and [`to_graph`] convert flat [`mcts_engine::tree_store::TreeStore`] structures
//!   into well-styled directed graphs.
//! - **Image Rendering**: [`render_svg`], [`render_png`], [`write_svg_file`], and [`write_png_file`]
//!   invoke the system `dot` binary to export visual diagrams.
//! - **Principal Variation Highlighting**: Automatically identifies and highlights the best action trajectory
//!   through the tree.
//! - **Flexible Statistics Support**: Built-in [`EdgeStatsView`] formatters for [`mcts_engine::selection::MultiAgentPuctStats`]
//!   and [`mcts_engine::selection::ismcts::IsmctsStats`], plus customizable label hooks.

pub mod builder;
pub mod mdp;
pub mod options;
pub mod render;
pub mod stats;

#[cfg(test)]
mod tests;

pub use builder::build_dot_graph;
pub use mdp::{graph_env_to_dot, graph_env_to_graph, render_graph_env_svg};
pub use options::{NodeFilter, RankDir, TreeVisualizerOptions};
pub use render::{
    render_png, render_svg, to_dot, to_graph, write_dot_file, write_png_file, write_svg_file,
};
pub use stats::EdgeStatsView;
