//! Asynchronous self-play trajectory worker for TicTacToe.
//!
//! Reuses the generic zero-allocation [`mcts_onnx::run_selfplay_session`] runner.

use mcts_onnx::{SelfPlayConfig, run_selfplay_session};
use std::path::PathBuf;
use tictactoe::TicTacToeState;

fn legal_actions(s: &TicTacToeState) -> Vec<usize> {
    let mut legal = Vec::new();
    s.legal_actions(&mut legal);
    legal
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut config = SelfPlayConfig {
        spool_dir: PathBuf::from("./spool_tictactoe"),
        model_path: None,
        num_games: 50,
        num_sims: 50,
        parallel_games: 16,
        c_puct: 1.414,
        worker_id: 0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        game_id: 0,
        action_dim: 9,
        num_players: 2,
        chunk_size: 50,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--spool-dir" => {
                i += 1;
                config.spool_dir = PathBuf::from(&args[i]);
            }
            "--model-path" => {
                i += 1;
                config.model_path = Some(PathBuf::from(&args[i]));
            }
            "--games" => {
                i += 1;
                config.num_games = args[i].parse().unwrap();
            }
            "--sims" => {
                i += 1;
                config.num_sims = args[i].parse().unwrap();
            }
            "--c-puct" => {
                i += 1;
                config.c_puct = args[i].parse().unwrap();
            }
            "--worker-id" => {
                i += 1;
                config.worker_id = args[i].parse().unwrap();
            }
            "--parallel-games" | "--batch-size" => {
                i += 1;
                config.parallel_games = args[i].parse().unwrap();
            }
            _ => {}
        }
        i += 1;
    }

    if let Err(err) = run_selfplay_session::<TicTacToeState>(config, legal_actions) {
        eprintln!("[tictactoe-selfplay] Worker error: {err}");
        std::process::exit(1);
    }
}
