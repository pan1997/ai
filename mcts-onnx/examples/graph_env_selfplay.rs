//! Self-play generator on discrete `GraphEnv` using MCTS and `TabularModel`.
//!
//! Spools trajectory chunks (`traj_*.bin`) for training verification.

use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_envs::GraphEnv;
use mcts_onnx::{StepRecord, TabularModel, TrajectorySpooler};
use mcts_traits::{AgentDynamics, AgentId};
use rand::Rng;
use std::path::PathBuf;

struct EpisodeStep {
    state: u32,
    action_mask: Vec<u8>,
    policy_target: Vec<f32>,
    action_taken: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut spool_dir = PathBuf::from("./spool_graph");
    let mut model_path: Option<PathBuf> = None;
    let mut num_games = 50;
    let mut num_sims = 40;
    let chain_len = 4;

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
            _ => {}
        }
        i += 1;
    }

    let env = GraphEnv::chain(chain_len);
    let num_actions = env.num_actions;
    let num_players = 1;

    let model = if let Some(ref path) = model_path {
        if path.exists() {
            println!("Loading tabular model from {}", path.display());
            TabularModel::load_from_file(path).expect("failed to load model")
        } else {
            println!("Model file not found at {}, using uniform model", path.display());
            TabularModel::new(num_actions, num_players)
        }
    } else {
        TabularModel::new(num_actions, num_players)
    };

    let mut spooler = TrajectorySpooler::new(
        &spool_dir,
        0,
        25,
        99, // game_id
        1,  // channels
        1,  // height
        chain_len as u32,
        num_actions as u32,
        num_players as u32,
    )
    .expect("failed to create spooler");

    let selection = MultiAgentPuctSelection::<1> { c_puct: 1.414 };
    let backup = VectorBackup::<1>::new(1.0);
    let mut rng = rand::thread_rng();

    let mut total_steps = 0;

    for _game in 0..num_games {
        let mut s = env.initial();
        let mut episode_steps: Vec<EpisodeStep> = Vec::new();
        let mut step_count = 0;
        let max_steps = chain_len * 3;

        while !env.is_terminal(s) && step_count < max_steps {
            step_count += 1;
            let mut tree = TreeStore::with_capacity(1024, 1024, MultiAgentPuctStats::<1>::new());
            let root = tree.insert_root(AgentId(0));

            SequentialScheduler.search(
                &mut tree,
                &env,
                &model,
                &selection,
                &backup,
                root,
                &s,
                num_sims,
            );

            let mut policy_target = vec![0.0; num_actions];
            let mut total_visits = 0;

            for edge in tree.child_edges(root) {
                let action = *tree.edge_action(edge) as usize;
                let visits = tree.stats.visits[edge.as_usize()];
                policy_target[action] = visits as f32;
                total_visits += visits;
            }

            if total_visits > 0 {
                for p in &mut policy_target {
                    *p /= total_visits as f32;
                }
            } else {
                policy_target = vec![1.0 / num_actions as f32; num_actions];
            }

            // Pick action: with probability 0.15 explore randomly, else sample from visit distribution
            let action_taken = if rng.gen_bool(0.15) {
                let mut legal = Vec::new();
                env.actions(&s, &mut legal);
                legal[rng.gen_range(0..legal.len())]
            } else {
                // Weighted sample from policy_target
                let r: f32 = rng.gen_range(0.0..1.0);
                let mut cum = 0.0;
                let mut chosen = 0;
                for (a, &p) in policy_target.iter().enumerate() {
                    cum += p;
                    if r <= cum {
                        chosen = a as u32;
                        break;
                    }
                }
                chosen
            };

            let mask = env.action_mask_for(s);

            episode_steps.push(EpisodeStep {
                state: s,
                action_mask: mask,
                policy_target,
                action_taken,
            });

            let outcome = env.step(&mut s, &action_taken);
            if outcome.terminated {
                break;
            }
        }

        // Determine final return (1.0 if reached terminal state chain_len - 1, else 0.0)
        let final_return = if s == (chain_len - 1) as u32 { 1.0 } else { 0.0 };

        for step in episode_steps {
            total_steps += 1;
            let obs = env.observation_for(step.state);
            let record = StepRecord {
                observation: obs,
                action_mask: step.action_mask,
                policy_target: step.policy_target,
                value_target: vec![final_return],
                action_taken: step.action_taken,
                reward_target: vec![final_return],
            };
            let _ = spooler.push(record).expect("spooler push failed");
        }
    }

    let flushed = spooler.flush().expect("spooler flush failed");
    if let Some(path) = flushed {
        println!("Flushed final chunk: {}", path.display());
    }
    println!(
        "Self-play complete: generated {} games, {} total steps into {}",
        num_games,
        total_steps,
        spool_dir.display()
    );
}
