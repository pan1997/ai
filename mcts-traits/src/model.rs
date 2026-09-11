/// Trait for inspecting scalar or multi-agent value estimates.
pub trait HasValue {
    fn value(&self) -> f32;
    fn values(&self) -> &[f32];
}

/// Trait for inspecting policy prior probabilities.
pub trait HasPolicy {
    fn priors(&self) -> &[f32];
}

/// Standard evaluation output containing policy priors and value estimates (scalar or multi-agent).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Evaluation {
    pub priors: Vec<f32>,
    pub values: Vec<f32>,
}

impl Evaluation {
    #[inline]
    pub fn scalar(priors: Vec<f32>, value: f32) -> Self {
        Self {
            priors,
            values: vec![value],
        }
    }

    #[inline]
    pub fn vector(priors: Vec<f32>, values: Vec<f32>) -> Self {
        Self { priors, values }
    }
}

impl HasValue for Evaluation {
    #[inline]
    fn value(&self) -> f32 {
        self.values.first().copied().unwrap_or(0.0)
    }

    #[inline]
    fn values(&self) -> &[f32] {
        &self.values
    }
}

impl HasPolicy for Evaluation {
    #[inline]
    fn priors(&self) -> &[f32] {
        &self.priors
    }
}

/// Strategy and evaluation interface.
/// Evaluates whatever state space the search plans over (e.g. board state or latent vector).
pub trait Model<S> {
    /// Evaluates priors and values for state `s`.
    fn evaluate(&self, s: &S) -> Evaluation;

    /// Probability distribution over legal actions in state `s`.
    fn prior(&self, s: &S) -> Vec<f32> {
        self.evaluate(s).priors
    }

    /// Scalar value estimate from perspective of the acting agent.
    fn value(&self, s: &S) -> f32 {
        self.evaluate(s).value()
    }
}

/// Batched evaluation interface for amortizing neural network / GPU tensor inference.
pub trait BatchedModel<S>: Model<S> {
    fn evaluate_batch(&self, states: &[&S]) -> Vec<Evaluation>;
}
