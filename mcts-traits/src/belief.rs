//! Belief-state sampling and observation sequence abstractions for imperfect-information games.
//!
//! In imperfect-information environments (e.g. card games, partially observable MDPs),
//! agents do not observe the ground-truth world state directly. Instead, they receive
//! filtered observations or sequential event deltas.
//!
//! This module provides:
//! - [`ObservationSequence`]: An ordered history of observation events emitted over time.
//! - [`BeliefSampler`]: Abstraction for sampling ground-truth states conditioned on an observation context.
//! - [`IncrementalBeliefSampler`]: Extension trait for samplers that incrementally update beliefs from deltas.

/// An ordered sequence of observation events received by an agent over time.
///
/// Encapsulates history accumulation for environments that emit incremental event deltas
/// rather than complete state snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObservationSequence<Obs> {
    history: Vec<Obs>,
}

impl<Obs> ObservationSequence<Obs> {
    /// Creates a new, empty observation sequence.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    /// Pre-allocates an observation sequence with the given capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: Vec::with_capacity(capacity),
        }
    }

    /// Appends a new observation event to the sequence.
    #[inline]
    pub fn push(&mut self, obs: Obs) {
        self.history.push(obs);
    }

    /// Extends the sequence by appending an iterator of observation events.
    #[inline]
    pub fn extend<I: IntoIterator<Item = Obs>>(&mut self, iter: I) {
        self.history.extend(iter);
    }

    /// Returns the most recently received observation event, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn latest(&self) -> Option<&Obs> {
        self.history.last()
    }

    /// Returns a slice of the accumulated observation history.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Obs] {
        &self.history
    }

    /// Returns the number of observation events recorded so far.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` if no observation events have been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Clears all recorded observation events.
    #[inline]
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Belief-state sampler for imperfect-information games.
///
/// Conditioned on `Context`, which can be:
/// - A full snapshot observation (e.g. `Context = SequenceObservation`), OR
/// - An accumulated history of events (e.g. `Context = ObservationSequence<Event>`), OR
/// - A custom belief distribution or particle filter.
///
/// Implementations must not impose blanket bounds (such as `Clone` or `Send`), enabling
/// zero-cost deterministic or stateful samplers.
pub trait BeliefSampler {
    /// The ground-truth state representation sampled for search trajectories.
    type State;
    /// The conditioning observation context (snapshot or history sequence).
    type Context;

    /// Samples a fresh ground-truth state conditioned on the observation context.
    fn sample(&mut self, context: &Self::Context) -> Self::State;
}

impl<B: BeliefSampler + ?Sized> BeliefSampler for &mut B {
    type State = B::State;
    type Context = B::Context;

    #[inline]
    fn sample(&mut self, context: &Self::Context) -> Self::State {
        (**self).sample(context)
    }
}

impl<B: BeliefSampler + ?Sized> BeliefSampler for Box<B> {
    type State = B::State;
    type Context = B::Context;

    #[inline]
    fn sample(&mut self, context: &Self::Context) -> Self::State {
        (**self).sample(context)
    }
}

/// Optional extension trait for belief samplers that incrementally update their belief context.
pub trait IncrementalBeliefSampler: BeliefSampler {
    /// Incremental event delta emitted by the environment.
    type Delta;

    /// Incorporates a new delta event into the observation context.
    fn update(&mut self, context: &mut Self::Context, delta: &Self::Delta);
}
