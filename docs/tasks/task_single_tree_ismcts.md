# Task: Implement True Single-Tree Information Set MCTS (Single-Tree ISMCTS)

**Issue / Task ID**: `TASK-MCTS-001`  
**Status**: Open  
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

## 4. Architectural Design & crate Boundaries

In accordance with [`AGENTS.md`](file:///home/pankaj/Projects/ai/AGENTS.md), changes must respect zero-allocation hot paths and modular crate boundaries:

### 4.1 `mcts-traits`: Determinization & State Sampler Abstraction
Define a minimal, zero-dependency trait for state belief sampling:

```rust
/// Belief-state sampler for imperfect-information games.
pub trait BeliefSampler {
    /// The ground-truth state representation sampled for search trajectories.
    type State;
    /// The filtered player-specific observation.
    type Observation;

    /// Samples a fresh ground-truth state conditioned on the observation.
    fn sample<R: rand::Rng>(&self, obs: &Self::Observation, rng: &mut R) -> Self::State;
}
```

### 4.2 `mcts-engine`: Availability Tracking & Selection Policy
1. **Edge Statistics**:
   Extend or parameterize SoA edge statistics in `TreeStore` to track `avail_visits: Vec<u32>`.
2. **Selection Policy**:
   Implement `IsmctsSelection` implementing `SelectionPolicy<Action, Reward, Stats>` that filters unexpanded or compatible edges based on current trajectory state compatibility.
3. **Scheduler**:
   Provide `IsmctsScheduler` (or extend `SequentialScheduler`) that:
   - Takes `sampler: &impl BeliefSampler`.
   - On each iteration $1..N$, calls `let mut state = sampler.sample(obs, &mut rng)`.
   - Traverses the tree matching `state.legal_actions()` against edge actions.
   - Increments $N_{\text{avail}}(s, a)$ for all available candidate actions at visited nodes.
   - Increments $N(s, a)$ for the traversed edge.
   - Expands and backs up.

### 4.3 `environments/sequence`: Agent Implementation
1. Implement `SequenceBeliefSampler`:
   Calls `sequence::dynamics::determinize_state(&obs, rng)`.
2. Create `SingleTreeIsMctsAgent`:
   Replaces or supplements `IsMctsAgent` with spec syntax `is-mcts-single:<iters>`.

---

## 5. Implementation Roadmap & Checklist

- [ ] **Phase 1: `mcts-traits` Abstractions**
  - [ ] Add `BeliefSampler` trait to `mcts-traits/src/dynamics.rs` or a dedicated `mcts-traits/src/belief.rs` module.
  - [ ] Add documentation and doc-tests with zero blanket trait bounds.

- [ ] **Phase 2: `mcts-engine` Core Engine Updates**
  - [ ] Add `IsmctsStats` structure tracking both `visits: Vec<u32>` and `avail_visits: Vec<u32>` in Structure-of-Arrays (SoA) layout.
  - [ ] Implement `IsmctsSelection` with availability count scaling.
  - [ ] Implement `IsmctsScheduler` with per-trajectory determinization sampling.
  - [ ] Add unit tests verifying availability count accumulation and compatibility filtering.

- [ ] **Phase 3: Environment Integration**
  - [ ] Implement `SequenceBeliefSampler` in `environments/sequence/src/dynamics.rs`.
  - [ ] Implement `SingleTreeIsMctsAgent` in `environments/sequence/src/agent.rs`.
  - [ ] Update CLI parser to support `is-mcts-single:<iters>`.
  - [ ] Implement `KuhnBeliefSampler` in `mcts-envs/src/kuhn_poker.rs` as a reference verification testbed.

- [ ] **Phase 4: Benchmarking & Verification**
  - [ ] Benchmark `is-mcts-single:500` vs `is-mcts:500:10` (Multi-Tree) over 50 games.
  - [ ] Benchmark `is-mcts-single:500` vs `mcts:500` (Clairvoyant Oracle) over 50 games.
  - [ ] Ensure memory allocations on the selection hot path remain strictly zero.

---

## 6. Acceptance Criteria

1. **Depth Improvement**: At 500 iterations, `SingleTreeIsMctsAgent` reaches average search depth $\ge 5$ plies (compared to $\le 3$ plies for 10-tree Multi-Tree MCTS).
2. **Win-Rate Parity / Superiority**: In a 50-game head-to-head match against `mcts:500` (oracle), `SingleTreeIsMctsAgent` achieves $\ge 50\%$ win rate without accessing ground-truth opponent hands.
3. **Zero Allocations on Selection Path**: Pre-allocated buffers are used for trajectory state advancing and candidate compatibility masks.
4. **Clean Toolchain**:
   - `cargo check --workspace` passes.
   - `cargo clippy --workspace --all-targets -- -D warnings` passes with 0 warnings.
   - `cargo test --workspace` passes.

