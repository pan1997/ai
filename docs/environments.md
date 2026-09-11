# Reference Environments & Evaluators

The `mcts-envs` crate contains reference game environments and baseline heuristic evaluators designed for benchmarking, verifying search mechanics, and testing reinforcement learning algorithms.

---

## 1. Environment Catalog

### 1.1 Connect 4 (`Connect4Dynamics<R, C>`)

Implemented in [`mcts-envs/src/connect4.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/connect4.rs).

- **Type**: 2-Player Zero-Sum, Turn-Based, Perfect Information.
- **Board**: Parametric grid of size $R \times C$ (default: $6 \times 7$).
- **Action Space**: Column index $c \in \{0, \dots, C-1\}$.
- **Win Condition**: 4 identical tokens connected horizontally, vertically, or diagonally.
- **Traits Implemented**:
  - `AgentDynamics`: Internal state transitions and legal column generation.
  - `BatchedAgentDynamics`: Vectorized step execution.
  - `World`: Full 2-player match referee with simultaneous joint-action interface.
- **Reward Encoding**:
  - Red Win: `[+1.0, -1.0]`
  - Yellow Win: `[-1.0, +1.0]`
  - Draw: `[0.0, 0.0]`

---

### 1.2 Hex (`HexDynamics<N>`)

Implemented in [`mcts-envs/src/hex.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/hex.rs).

- **Type**: 2-Player Zero-Sum, Turn-Based, Perfect Information, No Draws.
- **Board**: Rhombus of hexagons of size $N \times N$ (default: $11 \times 11$).
- **Players**:
  - `Black`: Must connect the Top row to the Bottom row.
  - `White`: Must connect the Left column to the Right column.
- **Action Space**: Cell index $r \cdot N + c \in \{0, \dots, N^2 - 1\}$.
- **Win Detection**: Breadth-First Search (BFS) over hexagonal neighboring graph:
  $$\Delta = \{(-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0)\}$$
- **Traits Implemented**: `AgentDynamics`, `World`.
- **Reward Encoding**: Winner gets `+1.0`, loser gets `-1.0`.

---

### 1.3 2048 / Tzf8 (`Tzf8Dynamics`)

Implemented in [`mcts-envs/src/tzf8.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/tzf8.rs).

- **Type**: Single-Player, Stochastic MDP.
- **Board**: $4 \times 4$ integer tile grid.
- **Action Space**: 4 directional shifts: `Left`, `Right`, `Up`, `Down`.
- **Dynamics**:
  - Sliders push and merge identical adjacent tiles, accumulating merged values into the score.
  - After each valid slide, a new tile (value 2 with 90% probability, value 4 with 10% probability) spawns in a random empty cell via xorshift64 PRNG.
- **Reward**: Immediate score gained during tile merges ($r_t \ge 0$).
- **Traits Implemented**: `AgentDynamics` (Single-agent return accumulation $Q = r + \gamma V$).

---

### 1.4 Kuhn Poker (`KuhnWorld` & `KuhnAgentDynamics`)

Implemented in [`mcts-envs/src/kuhn_poker.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/kuhn_poker.rs).

- **Type**: 2-Player Zero-Sum, Imperfect Information, Sequential Game Theory Benchmark.
- **Deck**: 3 cards: Jack (0), Queen (1), King (2).
- **Dealing**: Each player receives 1 private card; 1 card is discarded unseen.
- **Ante**: 1 chip each into the pot.
- **Action Space**: `Check`, `Bet`, `Call`, `Fold`.
- **Separation of Concerns Demonstration**:
  - `KuhnWorld`: True referee state (`cards: [u8; 2]`, `pot: [f32; 2]`, betting `history`). Generates filtered `KuhnObservation` so players only observe their own card.
  - `KuhnAgentDynamics`: Internal determinization / belief state planning used by an agent to evaluate hypothetical lines against modeled opponent card assignments.

---

## 2. Baseline Evaluators

### 2.1 `RolloutEvaluator<D>`

Implemented in [`mcts-envs/src/evaluators/rollout.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/evaluators/rollout.rs).

Performs classical Monte Carlo rollouts:
- Plays uniform-random actions forward from the leaf state up to `max_depth` or termination.
- Repeats for `num_rollouts` trajectories and averages returns.
- Provides a ground-truth baseline without requiring neural network training.

### 2.2 `UniformRandomModel<D>`

Implemented in [`mcts-envs/src/evaluators/random.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/evaluators/random.rs).

- Assigns uniform prior probability across all legal actions:
  $$P(s, a) = \frac{1}{|\mathcal{A}(s)|}$$
- Assigns initial value estimate $0.0$ for all players.
- Useful for validating pure UCT search behavior and debugging tree expansion.

---

## 3. Implementing a Custom Environment

To integrate your own game into `mcts-engine`, follow these steps:

### Step 1: Define State and Action Types
```rust
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MyGameState {
    // board or world state
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MyAction {
    Move(usize),
}
```

### Step 2: Implement `AgentDynamics`
```rust
use mcts_traits::{AgentDynamics, Transition};

pub struct MyGameDynamics;

impl AgentDynamics for MyGameDynamics {
    type State = MyGameState;
    type Action = MyAction;
    type Reward = [f32; 2]; // For 2-player games

    fn initial(&self) -> Self::State {
        MyGameState { /* ... */ }
    }

    fn actions(&self, s: &Self::State) -> Vec<Self::Action> {
        // Return legal actions
        vec![]
    }

    fn step(&self, s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward> {
        // Transition to next state, evaluate terminal state and rewards
        let next_state = s;
        let reward = [0.0, 0.0];
        let terminated = false;
        Transition::new(next_state, reward, terminated)
    }
}
```

### Step 3: Connect to MCTS
Your custom dynamics can immediately plug into any scheduler (`SequentialScheduler`, `BatchedScheduler`, `MultiGameScheduler`) and selection/backup policy without modification!

