//! Rendering and file export functions for MCTS tree visualizations.

use std::fmt::Debug;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use graphviz_rust::cmd::{CommandArg, Format};
use graphviz_rust::dot_structures::Graph;
use graphviz_rust::exec;
use graphviz_rust::print;
use graphviz_rust::printer::PrinterContext;

use mcts_engine::tree_store::{EdgeStatsStore, NodeId, TreeStore};

use crate::builder::build_dot_graph;
use crate::options::TreeVisualizerOptions;
use crate::stats::EdgeStatsView;

/// Builds a Graphviz [`Graph`] abstract syntax tree from an MCTS [`TreeStore`].
pub fn to_graph<Action, Reward, Stats, StepDelta>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
) -> Graph
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    build_dot_graph(tree, root, options)
}

/// Generates a Graphviz DOT language string representing `tree` starting from `root`.
pub fn to_dot<Action, Reward, Stats, StepDelta>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
) -> String
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    let graph = build_dot_graph(tree, root, options);
    print(graph, &mut PrinterContext::default())
}

/// Renders `tree` into an SVG string using the local `dot` installation.
///
/// # Errors
///
/// Returns an [`io::Error`] if `dot` fails to execute or is not installed.
pub fn render_svg<Action, Reward, Stats, StepDelta>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
) -> io::Result<String>
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    let graph = build_dot_graph(tree, root, options);
    let bytes = exec(
        graph,
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Renders `tree` into PNG image bytes using the local `dot` installation.
///
/// # Errors
///
/// Returns an [`io::Error`] if `dot` fails to execute or is not installed.
pub fn render_png<Action, Reward, Stats, StepDelta>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
) -> io::Result<Vec<u8>>
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    let graph = build_dot_graph(tree, root, options);
    exec(
        graph,
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Png)],
    )
}

/// Writes the Graphviz DOT representation of `tree` to `path`.
///
/// # Errors
///
/// Returns an [`io::Error`] if writing to `path` fails.
pub fn write_dot_file<Action, Reward, Stats, StepDelta, P: AsRef<Path>>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
    path: P,
) -> io::Result<()>
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    let dot = to_dot(tree, root, options);
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(dot.as_bytes())
}

/// Renders `tree` and writes the resulting SVG to `path`.
///
/// # Errors
///
/// Returns an [`io::Error`] if rendering fails or writing to `path` fails.
pub fn write_svg_file<Action, Reward, Stats, StepDelta, P: AsRef<Path>>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
    path: P,
) -> io::Result<()>
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    let svg = render_svg(tree, root, options)?;
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(svg.as_bytes())
}

/// Renders `tree` and writes the resulting PNG image to `path`.
///
/// # Errors
///
/// Returns an [`io::Error`] if rendering fails or writing to `path` fails.
pub fn write_png_file<Action, Reward, Stats, StepDelta, P: AsRef<Path>>(
    tree: &TreeStore<Action, Reward, Stats, StepDelta>,
    root: NodeId,
    options: &TreeVisualizerOptions<Action, StepDelta>,
    path: P,
) -> io::Result<()>
where
    Action: Debug,
    Reward: Debug,
    Stats: EdgeStatsStore + EdgeStatsView,
    StepDelta: Debug + PartialEq + Default,
{
    let png_bytes = render_png(tree, root, options)?;
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(&png_bytes)
}
