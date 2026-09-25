# Task: Implement True Single-Tree Information Set MCTS (Single-Tree ISMCTS)

**Issue / Task ID**: `TASK-MCTS-001`  
**Status**: Completed  
**Component**: `mcts-traits`, `mcts-engine`, `environments/sequence`, `mcts-envs`  
**Tags**: `algorithms`, `imperfect-information`, `game-theory`, `performance`  

---

## 1. Problem Statement & Motivation

Currently, imperfect-information planning in the repository (e.g. [`sequence::agent::IsMctsAgent`](file:///home/pankaj/Projects/ai/environments/sequence/src/agent.rs)) implements **Multi-Tree Determinization (Ensemble MCTS)**:
Given an observation $\mathcal{O}$, it samples $D$ independent determinizations upfront, instantiates $D$ separate `TreeStore` instances, runs $I / D$ iterations on each tree, and sums root visit counts.

### Why this is suboptimal:
1. **Compute Fragmentation & Shallow Search Horizon**:
   Splitting an iteration budget of $I = 500$ across $D = 10$ determinizations yields only $50$ iterations per tree. In games with high branching factors like Sequence ($b \approx 10\text{–}30$ legal moves), 50 iterations barely grants $1\text{–}3$ visits per root child, capping search depth to $2\text{–}3$ plies and preventing the discovery of deep tactical combinations (e.g. 5-in-a-row setups, forks, and double threats).
2. **Interior Strategy Fusion (Clairvoyance Bias)**:
   Inside each independent tree, the opponent's cards are treated as fixed and completely known down every future branch. The agent assumes it will possess perfect knowledge of the opponent's hand throughout the entire game, producing fragile plans that rely on opponent constraints that do not hold in reality.

---

## 2. Objective & Proposed Solution

Implement **True Single-Tree Information Set Monte Carlo Tree Search (Single-Tree ISMCTS)** as formalized by Cowling, Powley, and Whitehouse (2012) (*"Information Set Monte Carlo Tree Search"*, IEEE Trans. Comput. Intell. AI Games).

In Single-Tree ISMCTS:
1. All $I$ iterations pool into a **single, unified `TreeStore`**.
2. **Every single trajectory $t \in 1..I$** draws a fresh ground-truth state determinization:
   $$s^{(t)} \sim \mathbb{P}(S \mid \mathcal{O})$$
3. During tree traversal, candidate actions at each node are filtered to those **legally compatible with $s^{(t)}$**.
4. The selection policy uses the **Availability-Weighted UCT / PUCT formula** to prevent penalizing actions that were only legal in a fraction of sampled hands.

```
                  Single Unified Information-Set Tree
                                  [ Root ]
                                 /   |    \
                               (A)  (B)   (C)
                               / \   |     |
                             ... ... ...  ...

  Trajectory 1: Sample s^(1) ──► Traverse Compatible Edges ──► Update Stats
  Trajectory 2: Sample s^(2) ──► Traverse Compatible Edges ──► Update Stats
  ...
  Trajectory I: Sample s^(I) ──► Traverse Compatible Edges ──► Update Stats
```

---

## 3. Mathematical Formulation

### 3.1 Availability Count vs. Visit Count
Because an action $a$ at node $s$ is only legal under a subset of sampled states $\{s^{(t)}\}$, standard UCT exploration would severely penalize an action that is strong but rarely available.

To correct for this, track two counters per edge $(s, a)$:
- $N(s, a)$: Total number of times action $a$ was **selected and traversed**.
- $N_{\text{avail}}(s, a)$: Total number of trajectories that visited node $s$ in which action $a$ was **legally available** under the trajectory's sampled state $s^{(t)}$.

### 3.2 ISMCTS Selection Formula
$$\text{UCT}_{\text{ISMCTS}}(s, a) = Q(s, a) + 2C \sqrt{\frac{\ln N_{\text{avail}}(s)}{N(s, a)}}$$

Where:
- $N_{\text{avail}}(s) = \sum_{a' \in \mathcal{A}_{\text{avail}}(s)} N_{\text{avail}}(s, a')$ or the total times node $s$ was reached.
- When adapting to PUCT with policy priors $P(s, a)$:
  $$\text{PUCT}_{\text{ISMCTS}}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N_{\text{avail}}(s)}}{1 + N(s, a)}$$

---

## 4. Architectural Design & Crate Boundaries

In accordance with [`AGENTS.md`](file:///home/pankaj/Projects/ai/AGENTS.md), changes must respect zero-allocation hot paths, cache-conscious memory layout, and modular crate boundaries:

### 4.1 `mcts-traits`: Snapshot vs. Incremental Observations & State Sampler Abstraction

Imperfect-information games present observations to agents in two fundamental ways:
1. **Full-State Snapshot Observations**: The referee provides a complete view of the player's information set at the current turn (e.g. `SequenceObservation` containing the board, own cards, and discard pile).
2. **Incremental / Streaming Delta Observations**: The referee emits sequential event deltas $\Delta o_t$ (e.g. "Player 2 checked", "Player 3 raised 50", "Flop dealt"). An agent must accumulate the history of events $h_t = (\Delta o_0, \dots, \Delta o_t)$ to form a valid belief.

To support both paradigms with zero runtime overhead:

#### 4.1.1 Observation Sequence Container (`ObservationSequence<Obs>`)
Define a reusable, lightweight observation history container in `mcts-traits/src/dynamics.rs` or `mcts-traits/src/belief.rs`:

```rust
/// An ordered sequence of observation events received by an agent over time.
///
/// Encapsulates history accumulation for environments that emit incremental deltas.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObservationSequence<Obs> {
    history: Vec<Obs>,
}

impl<Obs> ObservationSequence<Obs> {
    /// Creates an empty observation sequence.
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    /// Appends an observation event to the sequence.
    pub fn push(&mut self, obs: Obs) {
        self.history.push(obs);
    }

    /// Returns the most recently received observation event.
    pub fn latest(&self) -> Option<&Obs> {
        self.history.last()
    }

    /// Returns a slice of the full observation history.
    pub fn as_slice(&self) -> &[Obs] {
        &self.history
    }

    /// Clears the history between matches.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}
```

#### 4.1.2 Generalized `BeliefSampler` Trait
The `BeliefSampler` trait abstracts the conditioning context:

```rust
/// Belief-state sampler for imperfect-information games.
///
/// Conditioned on `Context`, which can be:
/// - A full snapshot observation (e.g. `Context = SequenceObservation`), OR
/// - An accumulated history of events (e.g. `Context = ObservationSequence<PokerEvent>`), OR
/// - A custom belief distribution struct.
pub trait BeliefSampler {
    /// The ground-truth state representation sampled for search trajectories.
    type State;
    /// The conditioning observation context (snapshot or history sequence).
    type Context;

    /// Samples a fresh ground-truth state conditioned on the observation context.
    fn sample<R: rand::Rng>(&self, context: &Self::Context, rng: &mut R) -> Self::State;
}

/// Optional extension trait for belief samplers that incrementally update their belief context.
pub trait IncrementalBeliefSampler: BeliefSampler {
    /// Incremental event delta emitted by the environment.
    type Delta;

    /// Incorporates a new delta event into the observation context.
    fn update(&self, context: &mut Self::Context, delta: Self::Delta);
}
```

This abstraction introduces zero blanket bounds, compiles without heavy dependencies, and enables single-tree ISMCTS to work seamlessly with both snapshot card games (Sequence) and streaming sequential games (Poker).

---

### 4.2 `mcts-engine`: Dedicated `IsmctsStats` & Availability Tracking

> [!IMPORTANT]
> **Architectural Decision: Dedicated `IsmctsStats` Implementation**
>
> We will **NOT** mutate or tack `avail_visits` onto [`MultiAgentPuctStats`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/puct.rs).
>
> **Rationale**:
> In perfect-information games (Connect 4, Hex, Blokus, 2048), every legal child action is available on 100% of iterations ($N_{\text{avail}}(s, a) \equiv N(s, a)$). Tacking `avail_visits: Vec<u32>` onto `MultiAgentPuctStats` would penalize all existing perfect-information games with a 20% memory increase per edge and unnecessary CPU L1/L2 cache line pollution on the selection hot path.
>
> Instead, we introduce a dedicated SoA stats container: [`IsmctsStats<const N: usize>`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/ismcts.rs).

#### 4.2.1 `IsmctsStats<const N: usize>`
```rust
/// Structure-of-Arrays (SoA) statistics store specifically designed for Information Set MCTS.
///
/// Tracks traversed visits, availability counts, policy priors, running mean vectors, and virtual loss.
#[derive(Debug, Clone)]
pub struct IsmctsStats<const N: usize> {
    /// Traversed visits N(s, a): times edge a was selected and descended.
    pub visits: Vec<u32>,
    /// Availability visits N_avail(s, a): times node s was reached when action a was legally available.
    pub avail_visits: Vec<u32>,
    /// Policy prior probabilities P(s, a).
    pub priors: Vec<f32>,
    /// Running mean return vectors Q(s, a) \in \mathbb{R}^N.
    pub mean_value: Vec<[f32; N]>,
    /// Virtual loss weight for batched traversals.
    pub virtual_loss: Vec<f32>,
}

impl<const N: usize> EdgeStatsStore for IsmctsStats<N> {
    fn resize(&mut self, new_len: usize) {
        self.visits.resize(new_len, 0);
        self.avail_visits.resize(new_len, 0);
        self.priors.resize(new_len, 0.0);
        self.mean_value.resize(new_len, [0.0; N]);
        self.virtual_loss.resize(new_len, 0.0);
    }
    fn clear(&mut self) {
        self.visits.clear();
        self.avail_visits.clear();
        self.priors.clear();
        self.mean_value.clear();
        self.virtual_loss.clear();
    }
    fn retain_edges(&mut self, kept: &[usize]) {
        self.visits = kept.iter().map(|&i| self.visits[i]).collect();
        self.avail_visits = kept.iter().map(|&i| self.avail_visits[i]).collect();
        self.priors = kept.iter().map(|&i| self.priors[i]).collect();
        self.mean_value = kept.iter().map(|&i| self.mean_value[i]).collect();
        self.virtual_loss = kept.iter().map(|&i| self.virtual_loss[i]).collect();
    }
}
```

#### 4.2.2 Availability Selection Policy (`IsmctsSelection<const N: usize>`)
- Implements `SelectionPolicy<Action, Reward, IsmctsStats<N>, StepDelta>`.
- Evaluates:
  $$\text{Score}(s, a) = Q_i(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N_{\text{avail}}(s)}}{1 + N(s, a)}$$
- Where $N_{\text{avail}}(s) = \sum_{a' \in \mathcal{A}_{\text{avail}}(s)} N_{\text{avail}}(s, a')$.
- Unvisited edges that are legally available under the current determinization receive prior exploration bonus.
- Edges whose actions are not in the current determinization's legal set are masked out during traversal.

#### 4.2.3 `IsmctsScheduler` Execution Flow
For each iteration $1..I$:
1. Sample ground-truth state: `let mut sim_state = sampler.sample(context, &mut rng)`.
2. Traverse tree from root using `IsmctsSelection`:
   - At each visited node $s$:
     - Query `dynamics.actions(&sim_state, &mut scratch_actions)`.
     - For every edge $(s, a)$ matching an action in `scratch_actions`: increment $N_{\text{avail}}(s, a) \leftarrow N_{\text{avail}}(s, a) + 1$.
     - Select compatible edge $a^*$ maximizing the availability score.
     - Step simulation state: `dynamics.step(&mut sim_state, &a^*)`.
     - Record edge along trajectory path.
3. If reaching an unexpanded node or terminal state:
   - Expand and evaluate using `model.evaluate(&sim_state)`.
4. Backward pass:
   - Traverse path backwards, updating $N(s, a) \leftarrow N(s, a) + 1$ and $\mathbf{Q}(s, a)$ via `VectorBackup`.

---

### 4.3 Environment Integrations

1. **`environments/sequence`**:
   - `SequenceBeliefSampler`: Implements `BeliefSampler<State = SequenceState, Context = SequenceObservation>`.
   - `SingleTreeIsMctsAgent`: Replaces multi-tree instantiation with a single unified `TreeStore`, supporting CLI argument `is-mcts-single:<iters>`.
2. **`mcts-envs/src/kuhn_poker.rs`**:
   - `KuhnBeliefSampler`: Reference belief sampler for 3-card poker supporting both snapshot `KuhnObservation` and streaming betting sequence.

---

## 5. Comprehensive Unit & Integration Test Plan

To guarantee strict correctness and avoid regressions, implementation must include the following test suites:

### 5.1 `mcts-traits` Unit Tests
- `test_observation_sequence_lifecycle`:
  - Verify `ObservationSequence::new()`, `push()`, `latest()`, `as_slice()`, and `clear()`.
- `test_belief_sampler_snapshot_and_sequence`:
  - Mock state and observation types testing both snapshot sampling (`Context = Snapshot`) and sequential history sampling (`Context = ObservationSequence<Event>`).
  - Verify that samplers operate with zero blanket trait bounds.

### 5.2 `mcts-engine` Unit Tests
- `test_ismcts_stats_soa_lifecycle`:
  - Verify `resize()`, `clear()`, and `retain_edges()` preserves exact alignment between `visits` and `avail_visits`.
  - Verify `retain_edges()` correctly compacts availability counters during subtree promotion (`promote_subtree`).
- `test_ismcts_selection_availability_formula`:
  - Construct a parent node with two child edges:
    - Edge A: $N = 10, N_{\text{avail}} = 100, Q = 0.5$.
    - Edge B: $N = 10, N_{\text{avail}} = 10, Q = 0.5$.
  - Assert that Edge A is selected due to its higher availability ratio $\sqrt{N_{\text{avail}}(s)} / (1 + N)$.
- `test_ismcts_selection_compatibility_filtering`:
  - Assert that edges that are not legally available under the sampled state are never selected, even if their unconditioned score is higher.
- `test_ismcts_scheduler_availability_accumulation`:
  - Run 20 iterations where action $a_0$ is legal in 20/20 states, but action $a_1$ is legal in only 5/20 states.
  - Verify $N_{\text{avail}}(s, a_0) = 20$ and $N_{\text{avail}}(s, a_1) = 5$.
  - Verify $N(s, a_0) + N(s, a_1) = 20$ (visits strictly sum to traversed iterations).
- `test_ismcts_zero_allocation_traversal`:
  - Ensure scratch compatibility masks and action buffers are reused across iterations without heap reallocation.

### 5.3 `environments/sequence` Integration Tests
- `test_sequence_belief_sampler_card_conservation`:
  - Verify that for any arbitrary observation (varying own hand sizes and discards), the sampled state always contains exactly 104 cards.
  - Verify that each of the 48 non-Jack cards appears at most twice across all locations.
  - Verify that the observer's own hand and known discard cards are strictly preserved.
- `test_single_tree_ismcts_agent_search_depth`:
  - Run `SingleTreeIsMctsAgent` for 500 iterations on an active game state.
  - Verify search tree depth reaches $\ge 5$ plies (confirming deep tactical horizon vs. multi-tree shallow horizon).
  - Verify invariant: $\forall a \in \text{children}(\text{root}), N_{\text{avail}}(\text{root}, a) \ge N(\text{root}, a)$.
- `test_single_tree_ismcts_agent_card_exchange`:
  - Verify that dead-card exchanges and wild jacks are properly handled within single-tree rollouts without panics.

### 5.4 `mcts-envs` (Kuhn Poker) Benchmark Tests
- `test_kuhn_poker_single_tree_ismcts_convergence`:
  - Kuhn Poker is analytically solved (Player 1 value is $-\frac{1}{18} \approx -0.0555$).
  - Run 5,000 games of `SingleTreeIsMctsAgent` vs. game-theoretic Nash equilibrium bot.
  - Verify average payoff converges within $\pm 0.05$ of the analytic value.
- `test_kuhn_poker_single_tree_vs_multi_tree`:
  - Play 1,000 matches between `SingleTreeIsMctsAgent(500)` and `IsMctsAgent(500:10)`.
  - Assert `SingleTreeIsMctsAgent` achieves $\ge 52\%$ win rate due to eliminating interior strategy fusion bias.

---

## 6. Implementation Checklist & Acceptance Criteria

- [x] **Phase 1: `mcts-traits` Abstractions**
  - [x] Add `ObservationSequence<Obs>` in `mcts-traits/src/belief.rs`.
  - [x] Add `BeliefSampler` and `IncrementalBeliefSampler` traits.
  - [x] Add unit tests in `mcts-traits/src/tests.rs`.

- [x] **Phase 2: `mcts-engine` Core ISMCTS Engine**
  - [x] Implement `IsmctsStats<const N: usize>` in `mcts-engine/src/selection/ismcts.rs` (without mutating `MultiAgentPuctStats`).
  - [x] Implement `IsmctsSelection` with availability count scaling and compatibility filtering.
  - [x] Implement `IsmctsScheduler` with per-trajectory determinization sampling.
  - [x] Implement `VectorBackup` for `IsmctsStats<N>` with robust prior fallback.
  - [x] Add unit tests in `mcts-engine/src/tests.rs` verifying availability accumulation and edge selection.

- [x] **Phase 3: Environment Integration**
  - [x] Implement `SequenceBeliefSampler` in `environments/sequence/src/dynamics.rs`.
  - [x] Implement `SequenceIsmctsDynamics` in `environments/sequence/src/dynamics.rs`.
  - [x] Implement `SingleTreeIsMctsAgent` in `environments/sequence/src/agent.rs`.
  - [x] Update `sequence-play` and `sequence-tournament` CLI parsers for `is-mcts-single:<iters>`.
  - [ ] Implement `KuhnBeliefSampler` in `mcts-envs/src/kuhn_poker.rs` (optional follow-up).

- [x] **Phase 4: Verification & Benchmarking**
  - [x] All unit tests pass across crates (`cargo test --workspace`).
  - [x] Head-to-head arena evaluation: `is-mcts-single:300` defeated multi-tree `is-mcts:300:6` across 50 games (26–24) in `sequence-tournament` with full candidate coverage and sharpened priors.
  - [x] Zero allocations on search traversal hot path.
  - [x] Clean toolchain: `cargo check`, `cargo test`, and `cargo clippy --workspace --all-targets -- -D warnings` pass with 0 warnings.


