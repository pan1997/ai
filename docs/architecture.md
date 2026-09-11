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
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>);
    fn step(&self, s: &mut Self::State, action: &Self::Action) -> StepOutcome<Self::Reward>;
}
```

- If a sequential scheduler only needs `D::State: Clone` for root initialization, it specifies that requirement in its own `where` clause.
- Because `step` mutates `s: &mut State` in-place, zero cloning occurs down the selection trajectory.
- Algorithms that never clone states (e.g. MuZero recurrent latent dynamics) are not penalized by artificial constraints.
- Enables no-std compatibility and custom allocators where desired.

---

## 5. Zero-Allocation Traversals & Subtree Promotion

### 5.1 In-Place Search Descent
During MCTS selection, a path of depth $D$ is traversed from the root to a leaf:
1. `let mut state = root_state.clone();` clones the state once per iteration pass.
2. At each selected edge, `dynamics.step(&mut state, action)` mutates `state` in-place.
3. Node expansion reuses a pre-allocated scratch buffer via `dynamics.actions(&state, &mut scratch_actions)`.
4. As a result, the entire traversal and expansion sequence performs **zero heap allocations** on hot paths.

### 5.2 Subtree Promotion (`promote_subtree`)
In iterative game play (self-play or tournament matches), discarding the entire search tree after choosing an action wastes computational effort. `TreeStore::promote_subtree(new_root)` promotes any child node to become the new root:
- Runs a breadth-first search (BFS) over reachable nodes in the subtree.
- Re-indexes reachable nodes and edges contiguously into memory starting at `NodeId(0)`.
- Automatically prunes all unreachable sibling branches, compacting memory vectors.
- Retains visit counts, running means, policy priors, and transition rewards.

---

## 6. Future Roadmap: Directed Acyclic Graphs (DAG) vs Trees

A natural extension to standard MCTS trees is the incorporation of **Transposition Tables** (Zobrist hashing), transforming the tree into a Directed Acyclic Graph (DAG) where identical board states reached via transposed move orders share a single node.

### Theoretical Benefits
- **Sample Efficiency**: Search passes exploring different permutations of the same moves (e.g. `e4 e5 Nf3 Nc6` vs `Nf3 Nc6 e4 e5`) accumulate visit counts and value estimates in a shared node.
- **Deeper Search Horizon**: Transposition detection avoids redundant expansions of previously evaluated game states.

### Architectural Trade-offs & Complexities
1. **Multi-Path Backpropagation & Dual-Counting**:
   In a strict tree, each leaf has a unique predecessor path back to the root. In a DAG, propagating values backwards can cause visit counts $N(s)$ and value returns $Q(s)$ to be multi-counted if not carefully tracked via path-aware weighting or DAG-MCTS algorithms.
2. **Virtual Loss Intersections**:
   In parallel or batched search (`BatchedScheduler`), virtual losses prevent threads from following identical paths. In a DAG, two distinct paths may reconverge at an internal transposition, requiring global edge-level synchronization.
3. **Graph Cycles in Reversible Games**:
   Games with reversible moves (e.g. Chess piece maneuvering) can produce directed cycles in the search graph, requiring three-fold repetition detection or path-dependent state hashing.
4. **Memory Compaction & Subtree Promotion**:
   In an SoA tree, `promote_subtree` is a linear BFS. In a DAG, nodes can have multiple parents, requiring topological sorting or reference counting for garbage collection.

`TreeStore` maintains a pure tree representation for predictable, zero-allocation cache performance, while preserving clear extension hooks for future transposition-table layers.

