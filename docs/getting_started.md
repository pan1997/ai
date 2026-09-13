# Getting Started with MCTS

This guide walks you through setting up and running Monte Carlo Tree Search across common planning and game playing workflows.

---

## 1. Setting Up Dependencies

In your application's `Cargo.toml`:

```toml
[dependencies]
# Core traits and search engine
mcts-traits = { path = "path/to/mcts-traits" }
mcts-engine = { path = "path/to/mcts-engine" }

# Game environments (choose as needed)
connect4    = { path = "path/to/connect4" }
blokus      = { path = "path/to/blokus" }
tzf8        = { path = "path/to/tzf8" }
hex         = { path = "path/to/hex" }
mcts-envs   = { path = "path/to/mcts-envs" }
```

---

## 2. Basic Sequential Search (Connect 4)

Here is a complete, minimal example running sequential MCTS on a Connect 4 board:

```rust
use mcts_traits::{AgentDynamics, AgentId, TurnBasedDynamics};
use mcts_engine::tree_store::TreeStore;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use connect4::{Connect4State, Connect4World};
use mcts_envs::evaluators::UniformRandomModel;

fn main() {
    // 1. Initialize Environment and Evaluator
    let env = TurnBasedDynamics::new(Connect4World::<6, 7>::new());
    let model = UniformRandomModel::new(env, 2);

    // 2. Configure Selection and Backup Policies
    let selection = MultiAgentPuctSelection::<2> { c_puct: 1.414 };
    let backup = VectorBackup::<2>::default(); // gamma = 1.0

    // 3. Allocate the Search Tree
    // Pre-allocating capacity ensures zero heap allocations during tree expansion
    let stats = MultiAgentPuctStats::<2>::new();
    let mut tree = TreeStore::with_capacity(2000, 14000, stats);

    // 4. Initialize Root Node
    let root = tree.insert_root(AgentId(0));
    let initial_state = Connect4State::new();

    // 5. Execute MCTS Iterations
    let scheduler = SequentialScheduler;
    scheduler.search(
        &mut tree,
        &env,
        &model,
        &selection,
        &backup,
        root,
        &initial_state,
        500, // 500 iterations
    );

    // 6. Extract Best Action Based on Visit Counts
    let mut best_action = None;
    let mut max_visits = 0;

    for edge in tree.child_edges(root) {
        let visits = tree.stats.visits[edge.as_usize()];
        let action = tree.edge_action(edge);
        let q_val = tree.stats.mean_value[edge.as_usize()][0];
        println!("Column {}: visits = {}, Q = {:.4}", action, visits, q_val);

        if visits > max_visits {
            max_visits = visits;
            best_action = Some(*action);
        }
    }

    println!("Recommended move: Column {:?}", best_action);
}
```

---

## 3. High-Throughput Batched Search

When integrating neural networks (e.g. PyTorch, ONNX, Candle, Burn), evaluating one leaf at a time is bottlenecked by CPU-GPU transfer overhead. Use `BatchedScheduler` to evaluate batches of leaves at once:

```rust
use mcts_engine::scheduler::BatchedScheduler;

// Batched search: batch size of 16, virtual loss weight of 1.0
let scheduler = BatchedScheduler::new(16, 1.0);

scheduler.search(
    &mut tree,
    &env,
    &batched_model,
    &selection,
    &backup,
    root,
    &initial_state,
    50, // 50 batches of 16 = 800 simulations
);
```

During each batch:
- 16 diverse paths are explored using virtual loss.
- Unique leaf states are deduplicated.
- A single batched neural evaluation is queried.
- All 16 paths are backpropagated and virtual loss is cleanly removed.

---

## 4. Multi-Game Vectorized Self-Play

For large-scale self-play generation, `MultiGameScheduler` executes MCTS searches across multiple distinct games concurrently:

```rust
use mcts_engine::scheduler::MultiGameScheduler;

const NUM_GAMES: usize = 32;

let mut trees: Vec<_> = (0..NUM_GAMES)
    .map(|_| TreeStore::with_capacity(1000, 7000, MultiAgentPuctStats::<2>::new()))
    .collect();

let roots: Vec<_> = trees.iter_mut().map(|t| t.insert_root(AgentId(0))).collect();
let states: Vec<_> = (0..NUM_GAMES).map(|_| Connect4State::<6, 7>::new()).collect();
let state_refs: Vec<_> = states.iter().collect();

let scheduler = MultiGameScheduler::new(NUM_GAMES);
scheduler.search(
    &mut trees,
    &roots,
    &state_refs,
    &env,
    &batched_model,
    &selection,
    &backup,
    200, // 200 iterations per tree
);
```

All 32 games advance their simulation sweeps in lockstep, amortizing neural evaluations and keeping GPU utilization high.

---

## 5. Running Benchmark Arenas & CLI Games

The repository provides ready-to-run interactive CLI players and tournament arenas across the game crates:

### Connect 4
```bash
# Play against MCTS in the terminal
cargo run --release -p connect4 --bin connect4-play

# Run a round-robin tournament
cargo run --release -p connect4 --bin connect4-tournament -- --players mcts:1000,mcts:5000,random --games 20
```

### Blokus
```bash
# Play interactive Blokus Duo in the terminal
cargo run --release -p blokus --bin blokus-play

# Run a 4-player Blokus Classic tournament
cargo run --release -p blokus --bin blokus-tournament -- --players heuristic,mcts-hr:500:2:15,mcts-hu:2500,mcts:5000 --games 20
```

### 2048 / Tzf8
```bash
# Interactive terminal 2048 with real-time MCTS move evaluations
cargo run --release -p tzf8 --bin tzf8-play

# Evaluate Expectimax agents and normalization strategies across 100 boards
cargo run --release -p tzf8 --bin tzf8-tournament -- --agents heuristic,mcts:1000,mcts-norm:1000,mcts-puct:1000 --games 100
```

### Hex
```bash
# Interactive terminal Hex player (rhombus board rendering)
cargo run --release -p hex --bin hex-play -- --board-size 11 --pie-rule

# Multi-agent round-robin tournament (balanced first-mover advantage)
cargo run --release -p hex --bin hex-tournament -- --board-size 11 --agents heuristic,mcts-h:500,mcts:500,random --games 10 --pie-rule
```

