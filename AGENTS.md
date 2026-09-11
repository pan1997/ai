# AGENTS.md — Contributor & AI Agent Guidelines

This document provides operational instructions, architectural invariants, and design principles for AI assistants and human developers contributing to this repository.

---

## 1. Project Overview & Boundaries

This repository is a high-performance, zero-allocation Monte Carlo Tree Search (MCTS) toolkit partitioned into decoupled crates:

```
mcts-traits/  --> Zero-dependency traits and interfaces (abstractions only)
mcts-engine/  --> Core Structure-of-Arrays (SoA) engine, selection, backup, and schedulers
mcts-envs/    --> Reference environments, benchmark games, and heuristic/rollout evaluators
connect4/     --> Dedicated Connect 4 game engine, MCTS agents, and interactive CLI players
blokus/       --> Dedicated Blokus (Duo & Classic) game engine, multi-agent MCTS agents, and CLI players
```

### Dependency Rules
1. `mcts-traits` **must never** depend on `mcts-engine`, `mcts-envs`, `connect4`, or `blokus`.
2. `mcts-engine` depends **only** on `mcts-traits` and minimal math/random crates (`rand`, `rand_distr`). It must never depend on `mcts-envs`, `connect4`, or `blokus`.
3. `mcts-envs` depends on `mcts-traits` and optionally `mcts-engine` (for testing and integration).
4. Environment crates (e.g. `connect4`, `blokus`) depend on `mcts-traits` and `mcts-engine`.

---

## 2. Architectural Invariants

### 2.1 Structure-of-Arrays (SoA) Memory Layout
- The search tree (`TreeStore`) avoids pointer chasing (`Box`, `Rc`, `Arc`, or raw pointers).
- Nodes and edges are represented by 32-bit typed integer handles: `NodeId(u32)` and `EdgeId(u32)`.
- All node/edge properties are stored in flat, contiguous vectors (`Vec<T>`).
- **Zero Allocations on Hot Paths**: Traversing existing nodes during selection or updating edge statistics must never allocate memory on the heap. Allocations only occur during node expansion (`TreeStore::expand_node`).

### 2.2 Minimal & Deferred Trait Bounds
- In `mcts-traits`, associated types on traits like `AgentDynamics` (`type State; type Action; type Reward;`) must **not** carry blanket bounds such as `Send + Sync + Clone + Debug`.
- Algorithms or schedulers that need cloning or thread safety must declare those bounds locally on their own structs or method signatures.

### 2.3 Separation of `World` vs `AgentDynamics`
- `World` models the impartial environment / referee (full hidden card deals, simultaneous actions, tournament drivers, observation generation).
- `AgentDynamics` models the internal hypothetical planning transition of an individual agent (belief-state determinization or state transitions).
- Never conflate external game state with internal search state.

### 2.4 Vectorized Multi-Agent Returns
- Use `VectorBackup<N>` where returns $Q \in \mathbb{R}^N$ are tracked per player explicitly.
- Avoid the classic single-float sign-flip negation pattern ($Q = -Q$) for multi-player games, as it breaks in $N > 2$ players, imperfect information, and non-zero-sum dynamics.

---

## 3. Build, Test, and Verification Workflows

Always verify changes using the following toolchain commands:

```bash
# Check compilation across all crates
cargo check --workspace

# Run all unit tests and integration tests
cargo test --workspace

# Run doc-tests specifically
cargo test --workspace --doc

# Generate rustdoc documentation without external dependency clutter
cargo doc --workspace --no-deps
```

> [!IMPORTANT]
> When executing shell commands via subagents or background tasks in sandboxed environments, always use `--workspace` flags to ensure changes in one crate do not break dependents.

---

## 4. Code & Documentation Conventions

- **Edition**: Rust 2024.
- **In-Code Documentation**: Every public type, trait, method, and module must include rustdoc comments (`///` and `//!`).
  - Document parameter meanings, return values, failure/panic conditions, and algorithmic complexity.
  - Where mathematical formulas are involved (e.g. UCT, PUCT, Gumbel noise, discounted reward accumulation), write the LaTeX or ASCII formula in the doc comment.
- **Error Handling & Invariants**:
  - Out-of-bounds access or unexpanded child access should trigger clear, descriptive assertion panics with context (e.g., `NodeId`, `EdgeId`).
  - Internal state transformations should preserve indexing invariants.

---

## 5. Extension Recipes

### Adding a New Selection Policy
1. Define a struct implementing `mcts_engine::selection::SelectionPolicy<Action, Reward, Stats>`.
2. Implement `select_child(&self, store: &TreeStore<Action, Reward, Stats>, node_id: NodeId) -> Option<EdgeId>`.
3. Read edge statistics directly from `store.stats` without heap allocation.

### Adding a New Backup Policy
1. Define a struct implementing `mcts_engine::backup::BackupPolicy<Action, Reward, Stats, Evaluation>`.
2. Implement `init_root` to seed root priors if required.
3. Implement `backup` to traverse `path: &[PathElement]` backwards, updating visit counts and value estimates.

### Adding a New Environment
1. Implement `mcts_traits::AgentDynamics`:
   - `initial(&self) -> Self::State`
   - `actions(&self, s: &Self::State) -> Vec<Self::Action>`
   - `step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward>`
2. (Optional) Implement `mcts_traits::BatchedAgentDynamics` for batched simulators.
3. (Optional) Implement `mcts_traits::World` if the game supports multi-player arbitration, hidden information, or tournament play.

