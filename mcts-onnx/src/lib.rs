//! # MCTS ONNX
//!
//! High-throughput ONNX Runtime inference, dynamic micro-batching, model hot-swapping,
//! and binary trajectory spooling for high-performance Monte Carlo Tree Search.

pub mod client;
pub mod dispatcher;
pub mod spool;
pub mod tabular;
pub mod watcher;
pub mod wire;

#[cfg(test)]
mod tests;

pub use client::OnnxModelClient;
pub use dispatcher::{BatcherConfig, EvalRequest, EvaluationRaw, InferenceDispatcher};
pub use ort;
pub use spool::TrajectorySpooler;
pub use tabular::TabularModel;
pub use watcher::start_model_watcher;
pub use wire::{ChunkHeader, HEADER_SIZE, MAGIC, PROTOCOL_VERSION, StepRecord};
