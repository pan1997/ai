//! Direct in-thread ONNX Runtime model evaluator for synchronous batched inference.

use flume::Receiver;
use mcts_traits::{BatchedModel, Evaluation, Model, TensorRepresentable, softmax_masked};
use ort::session::Session;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Synchronous in-process ONNX model evaluator supporting batched inference without channel hops.
pub struct DirectOnnxModel<S> {
    session: Arc<Mutex<Session>>,
    reload_rx: Receiver<PathBuf>,
    legal_actions_fn: Arc<dyn Fn(&S) -> Vec<usize> + Send + Sync>,
    _marker: PhantomData<S>,
}

impl<S> DirectOnnxModel<S> {
    /// Creates a new `DirectOnnxModel` wrapping the active `Session` and optional reload channel.
    pub fn new(
        session: Session,
        reload_rx: Receiver<PathBuf>,
        legal_actions_fn: impl Fn(&S) -> Vec<usize> + Send + Sync + 'static,
    ) -> Self {
        Self {
            session: Arc::new(Mutex::new(session)),
            reload_rx,
            legal_actions_fn: Arc::new(legal_actions_fn),
            _marker: PhantomData,
        }
    }

    /// Checks if a weight reload request has arrived on the channel and hot-swaps the session.
    pub fn check_reload(&self) {
        while let Ok(new_path) = self.reload_rx.try_recv() {
            match Session::builder().and_then(|mut b| b.commit_from_file(&new_path)) {
                Ok(new_session) => {
                    println!(
                        "[DirectOnnxModel] Successfully hot-swapped ONNX weights from {:?}",
                        new_path
                    );
                    if let Ok(mut lock) = self.session.lock() {
                        *lock = new_session;
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[DirectOnnxModel] Failed to hot-swap ONNX session from {:?}: {err}",
                        new_path
                    );
                }
            }
        }
    }
}

impl<S: TensorRepresentable + Sync> Model<S> for DirectOnnxModel<S> {
    fn evaluate(&self, s: &S) -> Evaluation {
        let evals = self.evaluate_batch(&[s]);
        evals.into_iter().next().unwrap_or_else(|| Evaluation {
            priors: Vec::new(),
            values: Vec::new(),
        })
    }
}

impl<S: TensorRepresentable + Sync> BatchedModel<S> for DirectOnnxModel<S> {
    fn evaluate_batch(&self, states: &[&S]) -> Vec<Evaluation> {
        if states.is_empty() {
            return Vec::new();
        }

        self.check_reload();

        let batch_size = states.len();
        let feature_size = S::CHANNELS * S::HEIGHT * S::WIDTH;
        let mut buffer = vec![0.0f32; batch_size * feature_size];

        for (i, &s) in states.iter().enumerate() {
            let offset = i * feature_size;
            s.encode_tensor(&mut buffer[offset..offset + feature_size]);
        }

        let input_shape = [batch_size, S::CHANNELS, S::HEIGHT, S::WIDTH];
        let input_val = match ort::value::Tensor::from_array((input_shape, buffer)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[DirectOnnxModel] Tensor conversion failed: {e}");
                return vec![Evaluation::scalar(Vec::new(), 0.0); batch_size];
            }
        };

        let mut session_guard = match self.session.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        let run_res = session_guard.run(ort::inputs![input_val]);
        let outputs = match run_res {
            Ok(outs) => outs,
            Err(err) => {
                eprintln!("[DirectOnnxModel] Inference run failed: {err}");
                return vec![Evaluation::scalar(Vec::new(), 0.0); batch_size];
            }
        };

        let (_policy_shape, policy_slice) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(err) => {
                eprintln!("[DirectOnnxModel] Policy extraction failed: {err}");
                return vec![Evaluation::scalar(Vec::new(), 0.0); batch_size];
            }
        };

        let (_value_shape, value_slice) = match outputs[1].try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(err) => {
                eprintln!("[DirectOnnxModel] Value extraction failed: {err}");
                return vec![Evaluation::scalar(Vec::new(), 0.0); batch_size];
            }
        };

        let policy_stride = policy_slice.len() / batch_size;
        let value_stride = value_slice.len() / batch_size;

        let mut results = Vec::with_capacity(batch_size);
        for (i, &s) in states.iter().enumerate() {
            let p_start = i * policy_stride;
            let p_end = p_start + policy_stride;
            let policy_logits = &policy_slice[p_start..p_end];

            let v_start = i * value_stride;
            let v_end = v_start + value_stride;
            let values = value_slice[v_start..v_end].to_vec();

            let legal = (self.legal_actions_fn)(s);
            let priors = softmax_masked(policy_logits, &legal);

            results.push(Evaluation { priors, values });
        }

        results
    }
}
