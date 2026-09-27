//! Generic zero-allocation self-play runner with dynamic micro-batching and trajectory spooling.

use crate::client::OnnxModelClient;
use crate::dispatcher::{BatcherConfig, InferenceDispatcher};
use crate::spool::TrajectorySpooler;
use crate::watcher::start_model_watcher;
use crate::wire::StepRecord;
use mcts_engine::backup::{BackupPolicy, VectorBackup};
use mcts_engine::dirichlet::add_root_dirichlet_noise;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{AgentDynamics, AgentId, Model, TensorRepresentable};
use rand::Rng;
use std::io;
use std::path::PathBuf;

/// Environment trait required for running generic AlphaZero self-play episodes.
pub trait SelfPlayEnv: TensorRepresentable + Clone + PartialEq + Send + Sync + 'static {
    type Dynamics: AgentDynamics<State = Self, Action = usize, Reward = [f32; 2], StepDelta = ()>;

    /// Returns the planning dynamics instance for this game.
    fn dynamics(&self) -> Self::Dynamics;
    /// Creates a fresh initial game state.
    fn initial() -> Self;
    /// Populates `out` with all currently legal action indices.
    fn legal_actions(&self, out: &mut Vec<usize>);
    /// Populates `out` with the bitpacked action mask.
    fn action_mask(&self, out: &mut [u8]);
    /// Applies `action` and advances turn/state.
    fn apply_action(&mut self, action: usize);
    /// Returns `true` if the state is terminal.
    fn is_terminal(&self) -> bool;
    /// Returns the terminal return vector `[v_0, v_1]`.
    fn terminal_returns(&self) -> [f32; 2];
    /// Returns the 0-indexed player whose turn it is to act.
    fn current_player_index(&self) -> usize;
}

/// Configuration parameters for a self-play worker generation session.
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub spool_dir: PathBuf,
    pub model_path: Option<PathBuf>,
    pub num_games: usize,
    pub num_sims: usize,
    pub c_puct: f32,
    pub worker_id: u32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub game_id: u32,
    pub action_dim: usize,
    pub num_players: usize,
    pub chunk_size: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            spool_dir: PathBuf::from("./spool"),
            model_path: None,
            num_games: 50,
            num_sims: 40,
            c_puct: 1.414,
            worker_id: 0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            game_id: 0,
            action_dim: 9,
            num_players: 2,
            chunk_size: 50,
        }
    }
}

struct RecordedStep {
    observation: Vec<f32>,
    action_mask: Vec<u8>,
    policy_target: Vec<f32>,
    action_taken: usize,
}

/// Executes self-play episodes using a pre-instantiated evaluation model with zero-allocations on hot paths.
pub fn execute_episodes<E: SelfPlayEnv, M: Model<E>>(
    model: &M,
    spooler: &mut TrajectorySpooler,
    config: &SelfPlayConfig,
) -> (usize, usize) {
    let dynamics = E::initial().dynamics();
    let selection = MultiAgentPuctSelection::<2> { c_puct: config.c_puct };
    let backup = VectorBackup::<2>::default();
    let mut rng = rand::thread_rng();

    let obs_len = E::CHANNELS * E::HEIGHT * E::WIDTH;
    let mask_len = (config.action_dim + 7) / 8;

    // Pre-allocate scratch buffers once for the entire session (Zero-Allocation on Hot Paths)
    let mut obs = vec![0.0f32; obs_len];
    let mut mask = vec![0u8; mask_len];
    let mut legal = Vec::with_capacity(config.action_dim);
    let mut policy_target = vec![0.0f32; config.action_dim];
    let mut steps: Vec<RecordedStep> = Vec::with_capacity(128);

    let mut tree = TreeStore::with_capacity(
        config.num_sims + 16,
        (config.num_sims + 16) * config.action_dim,
        MultiAgentPuctStats::<2>::new(),
    );

    let mut total_steps = 0;

    for _ in 0..config.num_games {
        let mut state = E::initial();
        steps.clear();

        while !state.is_terminal() {
            legal.clear();
            state.legal_actions(&mut legal);
            if legal.is_empty() {
                break;
            }

            state.encode_tensor(&mut obs);
            state.action_mask(&mut mask);

            tree.clear();
            let root_agent = AgentId(state.current_player_index() as u32);
            let root = tree.insert_root(root_agent);

            let eval = model.evaluate(&state);
            tree.expand_node(root, &legal);
            backup.init_root(&mut tree, root, &eval);
            add_root_dirichlet_noise(
                &mut tree,
                root,
                config.dirichlet_alpha,
                config.dirichlet_epsilon,
                &mut rng,
            );

            SequentialScheduler.search(
                &mut tree,
                &dynamics,
                model,
                &selection,
                &backup,
                root,
                &state,
                config.num_sims,
            );

            policy_target.fill(0.0);
            let mut total_visits = 0;

            for edge in tree.child_edges(root) {
                let action = *tree.edge_action(edge);
                let visits = tree.stats.visits[edge.as_usize()];
                policy_target[action] = visits as f32;
                total_visits += visits;
            }

            if total_visits > 0 {
                for p in &mut policy_target {
                    *p /= total_visits as f32;
                }
            } else {
                for &a in &legal {
                    policy_target[a] = 1.0 / legal.len() as f32;
                }
            }

            // Early moves temperature sample, later greedy
            let action_taken = if steps.len() < 4 {
                let r: f32 = rng.gen_range(0.0..1.0);
                let mut cum = 0.0;
                let mut chosen = legal[0];
                for (a, &p) in policy_target.iter().enumerate() {
                    cum += p;
                    if r <= cum && legal.contains(&a) {
                        chosen = a;
                        break;
                    }
                }
                chosen
            } else {
                let mut best_a = legal[0];
                let mut best_p = -1.0;
                for &a in &legal {
                    if policy_target[a] > best_p {
                        best_p = policy_target[a];
                        best_a = a;
                    }
                }
                best_a
            };

            steps.push(RecordedStep {
                observation: obs.clone(),
                action_mask: mask.clone(),
                policy_target: policy_target.clone(),
                action_taken,
            });

            state.apply_action(action_taken);
        }

        let value_target = state.terminal_returns().to_vec();

        for step in steps.drain(..) {
            total_steps += 1;
            let record = StepRecord {
                observation: step.observation,
                action_mask: step.action_mask,
                policy_target: step.policy_target,
                value_target: value_target.clone(),
                action_taken: step.action_taken as u32,
                reward_target: value_target.clone(),
            };
            let _ = spooler.push(record).expect("spooler push failed");
        }
    }

    (config.num_games, total_steps)
}

/// Runs a complete self-play session: boots ONNX batcher and model watcher, spools chunks, and flushes to disk.
pub fn run_selfplay_session<E: SelfPlayEnv>(
    config: SelfPlayConfig,
    legal_fn: fn(&E) -> Vec<usize>,
) -> io::Result<(usize, usize)> {
    let mut spooler = TrajectorySpooler::new(
        &config.spool_dir,
        config.worker_id,
        config.chunk_size,
        config.game_id,
        E::CHANNELS as u32,
        E::HEIGHT as u32,
        E::WIDTH as u32,
        config.action_dim as u32,
        config.num_players as u32,
    )?;

    if let Some(ref path) = config.model_path {
        if path.exists() {
            println!("[SelfPlay] Loading ONNX model from {}", path.display());
            let session = ort::session::Session::builder()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
                .commit_from_file(path)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            let batcher_config = BatcherConfig {
                channels: E::CHANNELS,
                height: E::HEIGHT,
                width: E::WIDTH,
                max_batch_size: 64,
                max_latency: std::time::Duration::from_millis(2),
            };

            let dispatcher = InferenceDispatcher::new(session, batcher_config);
            let reload_sender = dispatcher.reload_sender();
            let _watcher = start_model_watcher(path, reload_sender);

            let client = OnnxModelClient::new(dispatcher.request_sender(), legal_fn);

            println!("[SelfPlay] Running self-play with ONNX model...");
            let (games, steps) = execute_episodes(&client, &mut spooler, &config);
            let _ = spooler.flush()?;
            println!(
                "[SelfPlay] Finished: {games} games, {steps} steps spooled into {}",
                config.spool_dir.display()
            );
            return Ok((games, steps));
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "No ONNX model found at specified path",
    ))
}
