//! Generic AlphaZero agent powered by ONNX Runtime and MCTS.

use crate::client::OnnxModelClient;
use crate::dispatcher::{BatcherConfig, InferenceDispatcher};
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{Agent, AgentDynamics, AgentId, TensorRepresentable};
use std::path::Path;

/// AlphaZero MCTS agent evaluating tree leaves via an ONNX neural network model.
pub struct AlphaZeroAgent<S: TensorRepresentable, D: AgentDynamics<State = S, Action = usize>> {
    name: String,
    dynamics: D,
    num_simulations: usize,
    c_puct: f32,
    _dispatcher: InferenceDispatcher,
    client: OnnxModelClient<S>,
    tree: TreeStore<usize, [f32; 2], MultiAgentPuctStats<2>, ()>,
    legal_scratch: Vec<usize>,
}

impl<S, D> AlphaZeroAgent<S, D>
where
    S: TensorRepresentable + Clone + PartialEq + Send + Sync + 'static,
    D: AgentDynamics<State = S, Action = usize, Reward = [f32; 2], StepDelta = ()>,
{
    /// Creates an `AlphaZeroAgent` from an ONNX model file path.
    pub fn from_onnx_file(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        num_simulations: usize,
        c_puct: f32,
        action_dim: usize,
        dynamics: D,
        legal_actions_fn: fn(&S) -> Vec<usize>,
    ) -> Result<Self, String> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(format!("Model file not found: {}", path.display()));
        }

        let session = ort::session::Session::builder()
            .map_err(|e| format!("Failed to create ORT session builder: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model from {}: {e}", path.display()))?;

        let config = BatcherConfig {
            channels: S::CHANNELS,
            height: S::HEIGHT,
            width: S::WIDTH,
            max_batch_size: 64,
            max_latency: std::time::Duration::from_millis(2),
        };

        let dispatcher = InferenceDispatcher::new(session, config);
        let client = OnnxModelClient::new(dispatcher.request_sender(), legal_actions_fn);

        let tree = TreeStore::with_capacity(
            num_simulations + 16,
            (num_simulations + 16) * action_dim,
            MultiAgentPuctStats::<2>::new(),
        );

        Ok(Self {
            name: name.into(),
            dynamics,
            num_simulations,
            c_puct,
            _dispatcher: dispatcher,
            client,
            tree,
            legal_scratch: Vec::with_capacity(action_dim),
        })
    }
}

impl<S, D> Agent<S, usize> for AlphaZeroAgent<S, D>
where
    S: TensorRepresentable + Clone + PartialEq + Sync,
    D: AgentDynamics<State = S, Action = usize, Reward = [f32; 2], StepDelta = ()>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &S) -> usize {
        self.legal_scratch.clear();
        self.dynamics.actions(state, &mut self.legal_scratch);
        if self.legal_scratch.len() <= 1 {
            return self.legal_scratch.first().copied().unwrap_or(0);
        }

        self.tree.clear();
        let agent = self.dynamics.current_agent(state);
        let root = self.tree.insert_root(agent);

        let selection = MultiAgentPuctSelection::<2> { c_puct: self.c_puct };
        let backup = VectorBackup::<2>::default();

        SequentialScheduler.search(
            &mut self.tree,
            &self.dynamics,
            &self.client,
            &selection,
            &backup,
            root,
            state,
            self.num_simulations,
        );

        let mut best_action = self.legal_scratch[0];
        let mut max_visits = -1;

        for edge in self.tree.child_edges(root) {
            let visits = self.tree.stats.visits[edge.as_usize()];
            if visits as i64 > max_visits {
                max_visits = visits as i64;
                best_action = *self.tree.edge_action(edge);
            }
        }

        best_action
    }
}
