//! Client adapter connecting search worker threads to the dynamic micro-batcher.

use crate::dispatcher::{EvalRequest, EvaluationRaw};
use flume::Sender;
use mcts_traits::{BatchedModel, Evaluation, Model, TensorRepresentable, softmax_masked};
use std::marker::PhantomData;
use std::sync::Arc;

/// Client that implements [`Model`] and [`BatchedModel`] for search threads by querying
/// the dynamic micro-batcher over high-throughput lockless channels.
pub struct OnnxModelClient<S> {
    req_tx: Sender<EvalRequest>,
    legal_actions_fn: Arc<dyn Fn(&S) -> Vec<usize> + Send + Sync>,
    _marker: PhantomData<S>,
}

impl<S> Clone for OnnxModelClient<S> {
    fn clone(&self) -> Self {
        Self {
            req_tx: self.req_tx.clone(),
            legal_actions_fn: Arc::clone(&self.legal_actions_fn),
            _marker: PhantomData,
        }
    }
}

impl<S> OnnxModelClient<S> {
    /// Creates a new `OnnxModelClient` with the designated request queue and legal actions extractor.
    pub fn new(
        req_tx: Sender<EvalRequest>,
        legal_actions_fn: impl Fn(&S) -> Vec<usize> + Send + Sync + 'static,
    ) -> Self {
        Self {
            req_tx,
            legal_actions_fn: Arc::new(legal_actions_fn),
            _marker: PhantomData,
        }
    }
}

impl<S: TensorRepresentable + Sync> Model<S> for OnnxModelClient<S> {
    fn evaluate(&self, s: &S) -> Evaluation {
        let mut buf = vec![0.0f32; S::CHANNELS * S::HEIGHT * S::WIDTH];
        s.encode_tensor(&mut buf);

        let (resp_tx, resp_rx) = flume::bounded(1);
        self.req_tx
            .send(EvalRequest {
                features: buf,
                respond_to: resp_tx,
            })
            .expect("InferenceDispatcher dropped channel");

        let raw: EvaluationRaw = resp_rx.recv().expect("Inference worker dropped response");
        let legal = (self.legal_actions_fn)(s);
        let priors = softmax_masked(&raw.policy_logits, &legal);

        Evaluation {
            priors,
            values: raw.values,
        }
    }
}

impl<S: TensorRepresentable + Sync> BatchedModel<S> for OnnxModelClient<S> {
    fn evaluate_batch(&self, states: &[&S]) -> Vec<Evaluation> {
        if states.is_empty() {
            return Vec::new();
        }

        let mut channels = Vec::with_capacity(states.len());
        for &s in states {
            let mut buf = vec![0.0f32; S::CHANNELS * S::HEIGHT * S::WIDTH];
            s.encode_tensor(&mut buf);

            let (resp_tx, resp_rx) = flume::bounded(1);
            let _ = self.req_tx.send(EvalRequest {
                features: buf,
                respond_to: resp_tx,
            });
            channels.push((s, resp_rx));
        }

        let mut evals = Vec::with_capacity(states.len());
        for (s, rx) in channels {
            let raw = rx.recv().expect("Inference worker dropped response");
            let legal = (self.legal_actions_fn)(s);
            let priors = softmax_masked(&raw.policy_logits, &legal);
            evals.push(Evaluation {
                priors,
                values: raw.values,
            });
        }

        evals
    }
}
