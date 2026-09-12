# High-Performance Monte Carlo Tree Search (MCTS) Engine

A modular, zero-allocation, Structure-of-Arrays (SoA) Monte Carlo Tree Search library built in Rust. Designed for modern reinforcement learning and game-tree search paradigms, including **AlphaZero**, **MuZero**, **Gumbel AlphaZero**, and multi-agent game theory.

```
       +-------------------------------------------------------------+
       |             Game Environments & Benchmark Arenas            |
       |  - connect4:  Dedicated Connect 4 game engine, agents & CLI |
       |  - blokus:    Blokus Classic & Duo multi-agent engine & CLI |
       |  - mcts-envs: Hex, 2048, Kuhn Poker, baseline evaluators    |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                        mcts-engine                          |
       |  Structure-of-Arrays TreeStore, Schedulers, Selection &     |
       |  Backup policies (UCT, PUCT, Gumbel, Vector Backup)         |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                        mcts-traits                          |
       |  Zero-dependency core abstractions: AgentDynamics, Model,  |
       |  World, TurnBasedWorld, TurnBasedDynamics, AgentId          |
       +-------------------------------------------------------------+
```

---

## Key Highlights

- **Cache-Conscious Memory Layout**: Contiguous Structure-of-Arrays (`TreeStore`) with 32-bit index handles (`NodeId`, `EdgeId`) eliminates pointer chasing and heap re-allocations along the hot search path.
- **Decoupled Architecture**:
  - `mcts-traits`: Zero-dependency, unopinionated trait interfaces. No blanket trait bounds; algorithms dictate their own requirements.
  - `mcts-engine`: High-throughput selection, backup, and scheduling engines.
  - `connect4` & `blokus`: Dedicated production-grade game engines and tournament CLI players.
  - `mcts-envs`: Reference environments and baseline rollout models for rapid benchmarking and verification.
- **Modern Search Algorithms**:
  - **UCT**: Classic exploration/exploitation formula with configurable constant.
  - **PUCT & Virtual Loss**: AlphaZero-style predictor-directed selection with lockless/virtual-loss parallelism.
  - **Gumbel AlphaZero**: Danihelka et al. (2022) policy improvement via Gumbel perturbation at the root node.
  - **Multi-Agent Vector Backup**: $Q \in \mathbb{R}^N$ element-wise returns, naturally handling general-sum games and eliminating the classic negation bug class in zero-sum games.
- **Flexible Schedulers**:
  - `SequentialScheduler`: Standard single-thread root-to-leaf sweep.
  - `BatchedScheduler`: Virtual-loss-guided batching with leaf deduplication to saturate GPU tensor cores during neural net inference.
  - `MultiGameScheduler`: Vectorized self-play across multiple parallel trees using SIMD/batch-friendly steps.
- **Imperfect Information & Referee Separation**:
  - Strict distinction between `World` (impartial referee, hidden state, simultaneous turns) and `AgentDynamics` (agent-internal hypothetical reasoning and belief states).
  - Universal `TurnBasedDynamics<W>` adapter for perfect-information turn-based games (`TurnBasedWorld`).

---

## Repository Structure

| Crate | Path | Description |
|---|---|---|
| [`mcts-traits`](file:///home/pankaj/Projects/ai/mcts-traits) | `mcts-traits/` | Core abstractions: `AgentDynamics`, `BatchedAgentDynamics`, `Model`, `BatchedModel`, `World`, `TurnBasedWorld`, `TurnBasedDynamics`, `Evaluation`, `AgentId`. |
| [`mcts-engine`](file:///home/pankaj/Projects/ai/mcts-engine) | `mcts-engine/` | SoA `TreeStore`, `UctSelection`, `MultiAgentPuctSelection`, `GumbelPuctSelection`, `VectorBackup`, and schedulers. |
| [`connect4`](file:///home/pankaj/Projects/ai/connect4) | `connect4/` | Dedicated Connect 4 game engine, MCTS agents, and interactive CLI players (`connect4-play`, `connect4-tournament`). |
| [`blokus`](file:///home/pankaj/Projects/ai/blokus) | `blokus/` | Dedicated Blokus (Classic & Duo) engine, polyomino registry, heuristic models, and CLI players (`blokus-play`, `blokus-tournament`). |
| [`mcts-envs`](file:///home/pankaj/Projects/ai/mcts-envs) | `mcts-envs/` | Reference environments: Hex, 2048, Kuhn Poker, plus rollout and baseline uniform evaluators. |

---

## Documentation Index

Explore the detailed topic guides:

- [Architecture Guide](file:///home/pankaj/Projects/ai/docs/architecture.md): Deep dive into the tripartite separation, SoA memory design, and the `World` vs `AgentDynamics` model.
- [Afterstate & Macro-Dynamics RFC](file:///home/pankaj/Projects/ai/docs/afterstate_and_macro_dynamics.md): Design exploration of round-based macro-dynamics, Afterstate nodes (MuZero style), and pluggable opponent policies.
- [Algorithms & Mathematics](file:///home/pankaj/Projects/ai/docs/algorithms.md): Exact mathematical formulations of UCT, PUCT, Gumbel AlphaZero, Vector Backup, and batched schedulers.
- [Reference Environments](file:///home/pankaj/Projects/ai/docs/environments.md): Rules, board representations, action spaces, and imperfect-information dynamics.
- [Getting Started & Tutorials](file:///home/pankaj/Projects/ai/docs/getting_started.md): Practical code walkthrough for setting up an environment, configuring MCTS, and running search sweeps.
- [Agent & Contributor Guide (`AGENTS.md`)](file:///home/pankaj/Projects/ai/AGENTS.md): Conventions, developer workflows, and guidance for autonomous coding agents.

---

## Quick Start

Add the crates to your `Cargo.toml`:

```toml
[dependencies]
mcts-traits = { path = "../mcts-traits" }
mcts-engine = { path = "../mcts-engine" }
mcts-envs = { path = "../mcts-envs" }
```

Run a simple 100-iteration MCTS search on Connect 4:

```rust
use mcts_traits::{AgentDynamics, AgentId, TurnBasedDynamics};
use mcts_engine::tree_store::TreeStore;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use connect4::{Connect4State, Connect4World};
use mcts_envs::evaluators::UniformRandomModel;

fn main() {
    let env = TurnBasedDynamics::new(Connect4World::<6, 7>::new());
    let model = UniformRandomModel::new(env, 2);
    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.414 };
    let backup = VectorBackup::<2>::default();
    let stats = MultiAgentPuctStats::<2>::new();

    let mut tree = TreeStore::with_capacity(1000, 7000, stats);
    let root = tree.insert_root(AgentId(0));
    let initial_state = Connect4State::new();

    let scheduler = SequentialScheduler;
    scheduler.search(&mut tree, &env, &model, &selection, &backup, root, &initial_state, 100);

    println!("Search complete! Nodes explored: {}", tree.num_nodes());
}
```

---

## Building & Verification

```bash
# Build all crates
cargo build --workspace

# Run all unit and doc tests
cargo test --workspace

# Generate local HTML Rustdoc wiki
cargo doc --workspace --no-deps --open
```

