# Architectural Blueprint

This document details the architectural principles, memory layouts, and design choices governing this high-performance Monte Carlo Tree Search (MCTS) engine.

---

## 1. Tripartite Crate Hierarchy

The workspace is organized into three distinct, decoupled crates:

```
+-------------------------------------------------------------+
|                         mcts-envs                           |
|  - Connect 4 (Parametric 2-player zero-sum)                 |
|  - Hex (Parametric graph connectivity via BFS)              |
|  - 2048 / Tzf8 (Stochastic single-agent puzzle)             |
|  - Kuhn Poker (Imperfect-information game theory)           |
|  - RolloutEvaluator & UniformRandomModel                    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                        mcts-engine                          |
|  - TreeStore (Structure-of-Arrays contiguous memory)         |
|  - Selection: UCT, MultiAgentPUCT, GumbelAlphaZero          |
|  - Backup: SingleAgentBackup, VectorBackup                  |
|  - Schedulers: Sequential, Batched, MultiGame               |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                        mcts-traits                          |
|  - AgentDynamics, BatchedAgentDynamics, Transition          |
|  - Model, BatchedModel, Evaluation, HasValue, HasPolicy     |
|  - World, AgentId, GraphEnv                                 |
+-------------------------------------------------------------+
```

### Decoupling Rationale
- **Zero Heavy Dependencies in Traits**: `mcts-traits` compiles in milliseconds and has zero mandatory runtime dependencies. This allows external libraries, neural network backends (e.g. PyTorch / ONNX / Candle / Burn), or game simulators to integrate without pulling in search engine implementation details.
- **Engine Agnostic to Game Details**: `mcts-engine` knows nothing about grids, cards, or board games. It operates strictly on generic types `Action`, `Reward`, and `Stats`.
- **Interchangeable Environments**: `mcts-envs` provides clean, reference implementations of various game types (deterministic, stochastic, zero-sum, imperfect-information) to benchmark search algorithms.

---

## 2. Structure-of-Arrays (SoA) Memory Architecture

Standard MCTS implementations typically represent trees as graphs of heap-allocated node pointers (`Box<Node>`, `Arc<RwLock<Node>>`, or `Vec<Node>` where each node owns a vector of children). This approach suffers from:
1. **Cache Misses**: Traversing pointers dereferences disparate, fragmented regions of memory.
2. **Allocation Overhead**: Dynamic child vectors trigger repeated system allocator reallocations during tree expansion.
3. **Synchronization Bottlenecks**: Fine-grained node-level mutexes cripple throughput in multi-threaded search.

### The `TreeStore` Solution
`TreeStore<Action, Reward, Stats>` solves these issues by adopting a Structure-of-Arrays (SoA) layout with contiguous backing arrays and typed 32-bit handles (`NodeId`, `EdgeId`):

```
                        +----------------------+
                        |      TreeStore       |
                        +----------+-----------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
  +--------------+                                    +--------------+
  |  NodeArrays  |                                    |  EdgeArrays  |
  +--------------+                                    +--------------+
  | parent_edge  |: Vec<EdgeId>                       | action       |: Vec<Action>
  | first_child  |: Vec<EdgeId>                       | child_node   |: Vec<NodeId>
  | num_children |: Vec<u32>                          | reward       |: Vec<Option<Reward>>
  | agent        |: Vec<AgentId>                      +--------------+
  | status       |: Vec<NodeStatus>                          |
  +--------------+                                           v
                                                      +--------------+
                                                      |  StatsStore  |
                                                      +--------------+
                                                      | visits       |: Vec<u32>
                                                      | priors       |: Vec<f32>
                                                      | mean_value   |: Vec<[f32; N]>
                                                      | virtual_loss |: Vec<f32>
                                                      +--------------+
```

### Contiguity & Cache Locality
- When expanding a node with $k$ legal actions, $k$ contiguous edge slots are allocated in `EdgeArrays` and `StatsStore`.
- Traversing child edges (`store.child_edges(node)`) is a sequential scan over a contiguous slice:
  $$\text{EdgeId}(first), \text{EdgeId}(first + 1), \dots, \text{EdgeId}(first + k - 1)$$
- All visits and priors for sibling edges reside on the same or adjacent CPU cache lines, minimizing L1/L2 data cache misses during selection passes.

### Index Size & Memory Footprint
By using `u32` for `NodeId` and `EdgeId`, each tree can scale up to **4,294,967,295** nodes and edges while halving pointer sizes compared to 64-bit machine pointers. `NodeId::INVALID` and `EdgeId::INVALID` are represented by `u32::MAX`.

---

## 3. Separation of Concerns: `World` vs `AgentDynamics`

A central architectural innovation in `mcts-traits` is the clean conceptual separation between external game arbitration and internal planning:

```
                      +-----------------------------+
                      |         World Trait         |
                      |  - Impartial Referee        |
                      |  - Full True State (Hidden) |
                      |  - Multi-Player Step        |
                      +--------------+--------------+
                                     |
               observe(ws, player)   |
                                     v
                      +-----------------------------+
                      |     AgentDynamics Trait     |
                      |  - Agent's Internal Mind    |
                      |  - Belief State / Latent    |
                      |  - Single-Agent Transitions |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |         MCTS Engine         |
                      |  - Hypothetical Search      |
                      +-----------------------------+
```

### `World`
- Represents the impartial ground-truth reality.
- Maintains hidden cards, shuffle decks, simultaneous turns, and refereeing rules.
- Generates observations (`ws -> Observation`) tailored to each player (e.g. filtering out an opponent's hole cards).

### `AgentDynamics`
- Represents what an agent *hypothesizes* during search.
- For perfect-information games (Connect 4, Hex, 2048), `AgentDynamics::State` is identical to `World::WorldState`.
- For imperfect-information games (Poker, Kriegspiel, Hanabi), `AgentDynamics::State` represents a determinized hypothetical state or belief distribution over hidden cards.
- Allows Information Set MCTS (ISMCTS) and MuZero latent dynamics to plug in seamlessly.

---

## 4. Trait Design Philosophy: Minimal & Deferred Bounds

Many generic Rust libraries enforce bounds directly on associated types:

```rust
// AVOID: Overly restrictive blanket bounds
pub trait AgentDynamics {
    type State: Clone + Send + Sync + std::fmt::Debug + 'static;
}
```

This crate follows the standard library's philosophy of **minimal trait bounds**:

```rust
// RECOMMENDED: Zero unnecessary bounds
pub trait AgentDynamics {
    type State;
    type Action: Eq + Debug;
    type Reward;
    
    fn initial(&self) -> Self::State;
    fn actions(&self, s: &Self::State) -> Vec<Self::Action>;
    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward>;
}
```

- If a sequential scheduler only needs `D::State: Clone`, it specifies that requirement in its own `where` clause.
- If an algorithm never clones states (e.g. MuZero recurrent latent dynamics), it is not penalized by artificial constraints.
- Enables no-std compatibility and custom allocators where desired.

