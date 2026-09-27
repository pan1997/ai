//! Asynchronous self-play trajectory worker for TicTacToe.
//!
//! Generates MCTS self-play games and writes binary chunks (`traj_*.bin`) via [`TrajectorySpooler`].
//! Supports hot-swapping ONNX models in real time when new weights are exported by the learner.

use mcts_engine::backup::{BackupPolicy, VectorBackup};
use mcts_engine::dirichlet::add_root_dirichlet_noise;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_onnx::{
    BatcherConfig, InferenceDispatcher, OnnxModelClient, StepRecord, TrajectorySpooler,
    start_model_watcher,
};
use mcts_traits::{AgentId, Model, TensorRepresentable};
use rand::Rng;
use std::path::PathBuf;
use tictactoe::{Player, RolloutEvaluator, TicTacToeDynamics, TicTacToeState};

struct RecordedStep {
    observation: Vec<f32>,
    action_mask: Vec<u8>,
    policy_target: Vec<f32>,
    action_taken: usize,
}

fn execute_selfplay_episodes<M: Model<TicTacToeState>>(
    model: &M,
    spooler: &mut TrajectorySpooler,
    num_games: usize,
    num_sims: usize,
    c_puct: f32,
) -> (usize, usize) {
    let dynamics = TicTacToeDynamics;
    let selection = MultiAgentPuctSelection::<2> { c_puct };
    let backup = VectorBackup::<2>::default();
    let mut rng = rand::thread_rng();

    let mut total_steps = 0;

    for _ in 0..num_games {
        let mut state = TicTacToeState::new();
        let mut steps: Vec<RecordedStep> = Vec::new();

        while !state.is_terminal() {
            let mut legal = Vec::new();
            state.legal_actions(&mut legal);
            if legal.is_empty() {
                break;
            }

            let mut obs = vec![0.0f32; TicTacToeState::CHANNELS * TicTacToeState::HEIGHT * TicTacToeState::WIDTH];
            state.encode_tensor(&mut obs);

            let mut mask = vec![0u8; 4];
            state.action_mask(&mut mask);

            let mut tree = TreeStore::with_capacity(
                num_sims + 10,
                (num_sims + 10) * 9,
                MultiAgentPuctStats::<2>::new(),
            );
            let root_agent = AgentId(state.current_player.index() as u32);
            let root = tree.insert_root(root_agent);

            // Expand root and inject Dirichlet exploration noise for diverse self-play
            let eval = model.evaluate(&state);
            tree.expand_node(root, &legal);
            backup.init_root(&mut tree, root, &eval);
            add_root_dirichlet_noise(&mut tree, root, 0.3, 0.25, &mut rng);

            SequentialScheduler.search(
                &mut tree,
                &dynamics,
                model,
                &selection,
                &backup,
                root,
                &state,
                num_sims,
            );

            let mut policy_target = vec![0.0f32; 9];
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

            // Temperature / exploration: early moves sample from visit distribution, later greedy
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
                // Greedy argmax
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
                observation: obs,
                action_mask: mask,
                policy_target,
                action_taken,
            });

            state.apply_action(action_taken);
        }

        let winner = state.check_winner();
        let value_target = match winner {
            Some(Player::X) => vec![1.0f32, -1.0f32],
            Some(Player::O) => vec![-1.0f32, 1.0f32],
            None => vec![0.0f32, 0.0f32],
        };

        for step in steps {
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

    (num_games, total_steps)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut spool_dir = PathBuf::from("./spool_tictactoe");
    let mut model_path: Option<PathBuf> = None;
    let mut num_games = 50;
    let mut num_sims = 50;
    let mut c_puct = 1.414;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--spool-dir" => {
                i += 1;
                spool_dir = PathBuf::from(&args[i]);
            }
            "--model-path" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--games" => {
                i += 1;
                num_games = args[i].parse().unwrap();
            }
            "--sims" => {
                i += 1;
                num_sims = args[i].parse().unwrap();
            }
            "--c-puct" => {
                i += 1;
                c_puct = args[i].parse().unwrap();
            }
            _ => {}
        }
        i += 1;
    }

    let mut spooler = TrajectorySpooler::new(
        &spool_dir,
        0,
        50,
        0, // game_id: TicTacToe
        TicTacToeState::CHANNELS as u32,
        TicTacToeState::HEIGHT as u32,
        TicTacToeState::WIDTH as u32,
        9, // action_dim
        2, // num_players
    )
    .expect("failed to create spooler");

    if let Some(ref path) = model_path {
        if path.exists() {
            println!("Loading ONNX model from {}", path.display());
            let session = mcts_onnx::ort::session::Session::builder()
                .expect("failed to create session builder")
                .commit_from_file(path)
                .expect("failed to load ONNX model");

            let config = BatcherConfig {
                channels: TicTacToeState::CHANNELS,
                height: TicTacToeState::HEIGHT,
                width: TicTacToeState::WIDTH,
                max_batch_size: 64,
                max_latency: std::time::Duration::from_millis(2),
            };

            let dispatcher = InferenceDispatcher::new(session, config);
            let reload_sender = dispatcher.reload_sender();
            let _watcher = start_model_watcher(path, reload_sender);

            let client = OnnxModelClient::new(dispatcher.request_sender(), |s: &TicTacToeState| {
                let mut legal = Vec::new();
                s.legal_actions(&mut legal);
                legal
            });

            println!("Running self-play with ONNX model...");
            let (games, steps) = execute_selfplay_episodes(&client, &mut spooler, num_games, num_sims, c_puct);
            let _ = spooler.flush().expect("flush failed");
            println!("Selfplay finished: {games} games, {steps} steps spooled into {}", spool_dir.display());
            return;
        }
    }

    println!("No ONNX model available, using RolloutEvaluator for self-play bootstrapping...");
    let evaluator = RolloutEvaluator::new(10, 10);
    let (games, steps) = execute_selfplay_episodes(&evaluator, &mut spooler, num_games, num_sims, c_puct);
    let _ = spooler.flush().expect("flush failed");
    println!("Bootstrap selfplay finished: {games} games, {steps} steps spooled into {}", spool_dir.display());
}
