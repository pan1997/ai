/// Trait for inspecting scalar or multi-agent value estimates.
pub trait HasValue {
    /// Returns the primary scalar value (e.g. value for the active agent or single player).
    fn value(&self) -> f32;

    /// Returns a slice of multi-agent value estimates across all participating agents.
    fn values(&self) -> &[f32];
}

/// Trait for inspecting policy prior probabilities.
pub trait HasPolicy {
    /// Returns the normalized policy prior distribution over legal actions.
    fn priors(&self) -> &[f32];
}

/// Standard evaluation output containing policy priors and value estimates (scalar or multi-agent).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Evaluation {
    /// Prior probability distribution over the available legal actions.
    pub priors: Vec<f32>,
    /// Value estimate vector across agents (length 1 for single-agent / scalar value).
    pub values: Vec<f32>,
}

impl Evaluation {
    /// Constructs an `Evaluation` for a single-agent or zero-sum scalar value setup.
    #[inline]
    pub fn scalar(priors: Vec<f32>, value: f32) -> Self {
        Self {
            priors,
            values: vec![value],
        }
    }

    /// Constructs an `Evaluation` for general multi-agent settings with per-agent value vector.
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

/// Strategy and evaluation interface for leaf states.
///
/// Evaluates whatever state space the search plans over (e.g. board state or latent vector).
/// Implementations include neural networks, random rollout policies, or heuristic estimators.
pub trait Model<S> {
    /// Evaluates both policy priors and value estimates for state `s`.
    fn evaluate(&self, s: &S) -> Evaluation;

    /// Returns the probability distribution over legal actions in state `s`.
    fn prior(&self, s: &S) -> Vec<f32> {
        self.evaluate(s).priors
    }

    /// Returns the scalar value estimate from the perspective of the acting agent.
    fn value(&self, s: &S) -> f32 {
        self.evaluate(s).value()
    }
}

impl<S, M: Model<S> + ?Sized> Model<S> for &M {
    #[inline]
    fn evaluate(&self, s: &S) -> Evaluation {
        (**self).evaluate(s)
    }

    #[inline]
    fn prior(&self, s: &S) -> Vec<f32> {
        (**self).prior(s)
    }

    #[inline]
    fn value(&self, s: &S) -> f32 {
        (**self).value(s)
    }
}

/// Batched evaluation interface for amortizing neural network / GPU tensor inference.
pub trait BatchedModel<S>: Model<S> {
    /// Evaluates a slice of state references in a single batched pass.
    ///
    /// The returned vector contains evaluations matching the order of input states.
    fn evaluate_batch(&self, states: &[&S]) -> Vec<Evaluation>;
}
