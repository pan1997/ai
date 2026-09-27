//! Generic zero-allocation self-play runner with MultiGameScheduler and trajectory spooling.

use crate::direct::DirectOnnxModel;
use crate::spool::TrajectorySpooler;
use crate::watcher::start_model_watcher;
use crate::wire::StepRecord;
use mcts_engine::backup::{BackupPolicy, VectorBackup};
use mcts_engine::dirichlet::add_root_dirichlet_noise;
use mcts_engine::scheduler::MultiGameScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::{NodeId, TreeStore};
use mcts_traits::{AgentId, BatchedAgentDynamics, BatchedModel, TensorRepresentable};
use rand::Rng;
use std::io;
use std::path::PathBuf;

/// Environment trait required for running generic AlphaZero self-play episodes.
pub trait SelfPlayEnv: TensorRepresentable + Clone + PartialEq + Send + Sync + 'static {
    type Dynamics: BatchedAgentDynamics<State = Self, Action = usize, Reward = [f32; 2], StepDelta = ()>;

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
    pub parallel_games: usize,
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
            parallel_games: 16,
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

/// Executes self-play episodes using `MultiGameScheduler` across parallel games with zero-allocations on hot paths.
pub fn execute_episodes<E: SelfPlayEnv, M: BatchedModel<E>>(
    model: &M,
    spooler: &mut TrajectorySpooler,
    config: &SelfPlayConfig,
) -> (usize, usize) {
    if config.num_games == 0 {
        return (0, 0);
    }

    let dynamics = E::initial().dynamics();
    let selection = MultiAgentPuctSelection::<2> { c_puct: config.c_puct };
    let backup = VectorBackup::<2>::default();
    let mut rng = rand::thread_rng();

    let obs_len = E::CHANNELS * E::HEIGHT * E::WIDTH;
    let mask_len = ((config.action_dim + 31) / 32) * 4;
    let batch_capacity = config.parallel_games.min(config.num_games).max(1);

    // Pre-allocate storage for `batch_capacity` concurrent games
    let mut trees: Vec<TreeStore<usize, [f32; 2], MultiAgentPuctStats<2>, ()>> = (0..batch_capacity)
        .map(|_| {
            TreeStore::with_capacity(
                config.num_sims + 16,
                (config.num_sims + 16) * config.action_dim,
                MultiAgentPuctStats::<2>::new(),
            )
        })
        .collect();

    let mut states: Vec<E> = (0..batch_capacity).map(|_| E::initial()).collect();
    let mut histories: Vec<Vec<RecordedStep>> = (0..batch_capacity)
        .map(|_| Vec::with_capacity(128))
        .collect();

    let mut roots: Vec<NodeId> = vec![NodeId(0); batch_capacity];
    let mut scratch_legal: Vec<Vec<usize>> = (0..batch_capacity)
        .map(|_| Vec::with_capacity(config.action_dim))
        .collect();
    let mut obs_scratch = vec![0.0f32; obs_len];
    let mut mask_scratch = vec![0u8; mask_len];
    let mut policy_scratch = vec![0.0f32; config.action_dim];

    let mut active_count = batch_capacity;
    let mut episodes_started = batch_capacity;
    let mut episodes_completed = 0;
    let mut total_steps = 0;

    while episodes_completed < config.num_games && active_count > 0 {
        // 1. Check for terminal states and flush completed trajectories to spooler.
        let mut b = 0;
        while b < active_count {
            if states[b].is_terminal() {
                let terminal_returns = states[b].terminal_returns().to_vec();
                for step in histories[b].drain(..) {
                    total_steps += 1;
                    let record = StepRecord {
                        observation: step.observation,
                        action_mask: step.action_mask,
                        policy_target: step.policy_target,
                        value_target: terminal_returns.clone(),
                        action_taken: step.action_taken as u32,
                        reward_target: terminal_returns.clone(),
                    };
                    let _ = spooler.push(record).expect("spooler push failed");
                }
                episodes_completed += 1;

                if episodes_started < config.num_games {
                    // Replenish slot with a fresh game
                    states[b] = E::initial();
                    trees[b].clear();
                    episodes_started += 1;
                    b += 1;
                } else {
                    // No more games to start; swap with last active slot (dense compaction)
                    active_count -= 1;
                    if b < active_count {
                        states.swap(b, active_count);
                        trees.swap(b, active_count);
                        histories.swap(b, active_count);
                        roots.swap(b, active_count);
                        scratch_legal.swap(b, active_count);
                    }
                }
            } else {
                b += 1;
            }
        }

        if active_count == 0 || episodes_completed >= config.num_games {
            break;
        }

        // 2. Setup root nodes across active games
        let mut root_refs: Vec<&E> = Vec::with_capacity(active_count);
        for b in 0..active_count {
            trees[b].clear();
            let agent = AgentId(states[b].current_player_index() as u32);
            let root = trees[b].insert_root(agent);
            roots[b] = root;

            scratch_legal[b].clear();
            states[b].legal_actions(&mut scratch_legal[b]);
            trees[b].expand_node(root, &scratch_legal[b]);
            root_refs.push(&states[b]);
        }

        // 3. Batched evaluation of active root states in a single forward pass
        let root_evals = model.evaluate_batch(&root_refs);

        // 4. Initialize roots and inject Dirichlet exploration noise
        for b in 0..active_count {
            backup.init_root(&mut trees[b], roots[b], &root_evals[b]);
            add_root_dirichlet_noise(
                &mut trees[b],
                roots[b],
                config.dirichlet_alpha,
                config.dirichlet_epsilon,
                &mut rng,
            );
        }

        // 5. Run parallel simulation sweeps using MultiGameScheduler
        let scheduler = MultiGameScheduler::new(active_count);
        scheduler.search(
            &mut trees[0..active_count],
            &roots[0..active_count],
            &root_refs[0..active_count],
            &dynamics,
            model,
            &selection,
            &backup,
            config.num_sims,
        );

        // 6. Select and apply actions across all active games
        for b in 0..active_count {
            let root = roots[b];
            let legal = &scratch_legal[b];
            if legal.is_empty() {
                continue;
            }

            policy_scratch.fill(0.0);
            let mut total_visits = 0;
            for edge in trees[b].child_edges(root) {
                let action = *trees[b].edge_action(edge);
                let visits = trees[b].stats.visits[edge.as_usize()];
                policy_scratch[action] = visits as f32;
                total_visits += visits;
            }

            if total_visits > 0 {
                for p in &mut policy_scratch {
                    *p /= total_visits as f32;
                }
            } else {
                for &a in legal {
                    policy_scratch[a] = 1.0 / legal.len() as f32;
                }
            }

            // Temperature sampling for first 4 moves of this game, then argmax
            let action_taken = if histories[b].len() < 4 {
                let r: f32 = rng.gen_range(0.0..1.0);
                let mut cum = 0.0;
                let mut chosen = legal[0];
                for (a, &p) in policy_scratch.iter().enumerate() {
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
                for &a in legal {
                    if policy_scratch[a] > best_p {
                        best_p = policy_scratch[a];
                        best_a = a;
                    }
                }
                best_a
            };

            states[b].encode_tensor(&mut obs_scratch);
            states[b].action_mask(&mut mask_scratch);

            histories[b].push(RecordedStep {
                observation: obs_scratch.clone(),
                action_mask: mask_scratch.clone(),
                policy_target: policy_scratch.clone(),
                action_taken,
            });

            states[b].apply_action(action_taken);
        }
    }

    (episodes_completed, total_steps)
}

/// Runs a complete self-play session: loads ONNX model, boots model watcher, spools chunks, and flushes to disk.
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

            let (reload_tx, reload_rx) = flume::unbounded();
            let _watcher = start_model_watcher(path, reload_tx);

            let direct_model = DirectOnnxModel::new(session, reload_rx, legal_fn);

            println!(
                "[SelfPlay] Running self-play with MultiGameScheduler ({} parallel games)...",
                config.parallel_games
            );
            let (games, steps) = execute_episodes(&direct_model, &mut spooler, &config);
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
