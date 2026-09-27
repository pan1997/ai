//! Asynchronous dynamic micro-batching inference dispatcher backed by ONNX Runtime.

use flume::{Receiver, Sender};
use ort::session::Session;
use std::path::PathBuf;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Raw neural network evaluation output across the full action and player spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationRaw {
    /// Action policy logits over the full action domain $|\mathcal{A}|$.
    pub policy_logits: Vec<f32>,
    /// Value estimate vector across all $N$ participating players.
    pub values: Vec<f32>,
}

/// Request sent from search actor threads to the dynamic micro-batcher.
pub struct EvalRequest {
    /// Flattened planar observation features ($C \times H \times W$).
    pub features: Vec<f32>,
    /// One-shot response channel back to the calling search thread.
    pub respond_to: Sender<EvaluationRaw>,
}

/// Configuration parameters for the dynamic micro-batching dispatcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Channels of the input tensor ($C$).
    pub channels: usize,
    /// Height of the input tensor ($H$).
    pub height: usize,
    /// Width of the input tensor ($W$).
    pub width: usize,
    /// Maximum batch size ($B_{\text{max}}$) before eagerly flushing to hardware.
    pub max_batch_size: usize,
    /// Maximum wait latency before flushing an incomplete micro-batch.
    pub max_latency: Duration,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            channels: 2,
            height: 3,
            width: 3,
            max_batch_size: 64,
            max_latency: Duration::from_millis(2),
        }
    }
}

/// Dynamic micro-batching inference dispatcher managing a dedicated worker thread.
pub struct InferenceDispatcher {
    req_tx: Sender<EvalRequest>,
    reload_tx: Sender<PathBuf>,
    shutdown_tx: Sender<()>,
    worker_handle: Option<JoinHandle<()>>,
}

impl InferenceDispatcher {
    /// Spawns the inference dispatcher with the initial `ort::Session` and batching config.
    pub fn new(session: Session, config: BatcherConfig) -> Self {
        let (req_tx, req_rx) = flume::unbounded();
        let (reload_tx, reload_rx) = flume::unbounded();
        let (shutdown_tx, shutdown_rx) = flume::unbounded();

        let handle = thread::spawn(move || {
            run_inference_loop(session, req_rx, reload_rx, shutdown_rx, config);
        });

        Self {
            req_tx,
            reload_tx,
            shutdown_tx,
            worker_handle: Some(handle),
        }
    }

    /// Returns a cloned sender channel to enqueue evaluation queries.
    pub fn request_sender(&self) -> Sender<EvalRequest> {
        self.req_tx.clone()
    }

    /// Returns a cloned sender channel to send model reload signals.
    pub fn reload_sender(&self) -> Sender<PathBuf> {
        self.reload_tx.clone()
    }

    /// Requests hot-swapping the active ONNX session from a new file path.
    pub fn reload_model(&self, path: PathBuf) -> Result<(), flume::SendError<PathBuf>> {
        self.reload_tx.send(path)
    }
}

impl Drop for InferenceDispatcher {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_inference_loop(
    mut session: Session,
    req_rx: Receiver<EvalRequest>,
    reload_rx: Receiver<PathBuf>,
    shutdown_rx: Receiver<()>,
    config: BatcherConfig,
) {
    let feature_size = config.channels * config.height * config.width;
    let mut request_batch = Vec::with_capacity(config.max_batch_size);
    let mut batch_features = Vec::with_capacity(config.max_batch_size * feature_size);

    loop {
        if shutdown_rx.try_recv().is_ok() {
            break;
        }

        let first_req = match req_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => req,
            Err(flume::RecvTimeoutError::Timeout) => {
                if shutdown_rx.try_recv().is_ok() {
                    break;
                }
                continue;
            }
            Err(flume::RecvTimeoutError::Disconnected) => break,
        };
        // 1. Check for weight reload requests
        if let Ok(new_path) = reload_rx.try_recv() {
            match Session::builder().and_then(|mut b| b.commit_from_file(&new_path)) {
                Ok(new_session) => {
                    session = new_session;
                    println!("[InferenceDispatcher] Successfully hot-swapped ONNX weights from {:?}", new_path);
                }
                Err(err) => {
                    eprintln!("[InferenceDispatcher] Failed to hot-swap ONNX session from {:?}: {err}", new_path);
                }
            }
        }

        // 2. Micro-batch dynamic accumulation
        request_batch.clear();
        request_batch.push(first_req);
        let deadline = Instant::now() + config.max_latency;

        while request_batch.len() < config.max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            if let Ok(req) = req_rx.recv_timeout(remaining) {
                request_batch.push(req);
            } else {
                break;
            }
        }

        // 3. Assemble contiguous NCHW array
        let current_batch_size = request_batch.len();
        batch_features.clear();
        for req in &request_batch {
            batch_features.extend_from_slice(&req.features);
        }
        // 4. Run batched ONNX session
        let input_shape = [current_batch_size, config.channels, config.height, config.width];
        let input_val = match ort::value::Tensor::from_array((input_shape, batch_features.clone())) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[InferenceDispatcher] Tensor conversion failed: {e}");
                continue;
            }
        };

        let run_res = session.run(ort::inputs![input_val]);
        let outputs = match run_res {
            Ok(outs) => outs,
            Err(err) => {
                eprintln!("[InferenceDispatcher] Inference run failed: {err}");
                continue;
            }
        };

        let (_policy_shape, policy_slice) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(err) => {
                eprintln!("[InferenceDispatcher] Policy extraction failed: {err}");
                continue;
            }
        };

        let (_value_shape, value_slice) = match outputs[1].try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(err) => {
                eprintln!("[InferenceDispatcher] Value extraction failed: {err}");
                continue;
            }
        };

        let policy_stride = policy_slice.len() / current_batch_size;
        let value_stride = value_slice.len() / current_batch_size;

        // 5. Fan-out responses along Axis(0)
        for (i, req) in request_batch.drain(..).enumerate() {
            let p_start = i * policy_stride;
            let p_end = p_start + policy_stride;
            let policy_logits = policy_slice[p_start..p_end].to_vec();

            let v_start = i * value_stride;
            let v_end = v_start + value_stride;
            let values = value_slice[v_start..v_end].to_vec();

            let _ = req.respond_to.send(EvaluationRaw { policy_logits, values });
        }
    }
}
