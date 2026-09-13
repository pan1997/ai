# Reference Environments & Evaluators

The `mcts-envs` crate contains reference game environments and baseline heuristic evaluators designed for benchmarking, verifying search mechanics, and testing reinforcement learning algorithms.

---

## 1. Environment Catalog

### 1.1 Connect 4 (`Connect4World<R, C>` & `TurnBasedDynamics<Connect4World<R, C>>`)

Implemented in [`connect4`](file:///home/pankaj/Projects/ai/connect4).

- **Type**: 2-Player Zero-Sum, Turn-Based, Perfect Information.
- **Board**: Parametric grid of size $R \times C$ (default: $6 \times 7$).
- **Action Space**: Column index $c \in \{0, \dots, C-1\}$.
- **Win Condition**: 4 identical tokens connected horizontally, vertically, or diagonally.
- **Traits Implemented**:
  - `World` & `TurnBasedWorld`: Ground-truth 2-player match referee with simultaneous joint-action interface and single-action step.
  - `TurnBasedDynamics<Connect4World>`: Reusable ego-centric planning dynamics.
  - `MacroConnect4Dynamics`: Round-based lookahead absorbing opponent policies.
- **Reward Encoding**:
  - Red Win: `[+1.0, -1.0]`
  - Yellow Win: `[-1.0, +1.0]`
  - Draw: `[0.0, 0.0]`

---

### 1.2 Hex (`hex` Crate)

Implemented in [`hex`](file:///home/pankaj/Projects/ai/hex) (with backwards-compatible re-exports in [`mcts-envs/src/hex.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/hex.rs)).

- **Type**: 2-Player Zero-Sum, Turn-Based, Perfect Information, Deterministic (No Draws by Hex Theorem).
- **Board**: Rhombus of hexagons with parametric dimension $N \times N$ (international tournament standard: $11 \times 11$).
- **Players**:
  - `Black`: Must connect the Top row to the Bottom row (moves first).
  - `White`: Must connect the Left column to the Right column (moves second).
- **Action Space**: Cell index $r \cdot N + c \in \{0, \dots, N^2 - 1\}$, algebraic notation (`A1` through `K11`), row-column pairs, or `HexAction::Swap` (Pie rule).
- **Pie (Swap) Rule**: Supported via `HexConfig { pie_rule: true }` and `--pie-rule` CLI flag. On Move 2, Player 2 (White) can choose to "swap" and take Black's opening move, effectively mitigating Hex's theoretical first-mover advantage.
- **Win Detection**: High-performance Disjoint Set Union (DSU) connectivity tracking with path compression and virtual boundary endpoints.
- **Evaluators**:
  - `ShortestPathHeuristicEvaluator`: 0-1 BFS / Dijkstra measuring the minimum stones each player needs to complete a connection ($d_{\text{Black}}$ vs $d_{\text{White}}$). Highly informative, sub-microsecond position evaluation.
  - `RolloutEvaluator`: Monte Carlo random rollouts to terminal states. Implements an optimized batch-fill permutation strategy that populates all remaining empty cells in one pass with pre-allocated scratch DSU buffers, executing hundreds of thousands of rollouts/sec.
  - `UniformEvaluator`: Fast uniform prior baseline.
- **Agents**:
  - `HeuristicAgent`: 1-ply greedy lookahead using the 0-1 BFS shortest path metric.
  - `MctsAgent`: Zero-copy MCTS planning supporting UCT, PUCT, rollout models, and heuristic guidance.
  - `RandomAgent`: Uniform random legal actions.
  - `HumanAgent`: Interactive keyboard player accepting algebraic coordinates (`F6`), row/col (`5 5`), or flat cell indices.
- **CLI Utilities**:
  - `hex-play`: Interactive terminal game with colored ANSI rhombus rendering, live move candidate inspection, and win detection. Options: `--board-size <N>` (alias `--size`), `--pie-rule`.
  - `hex-tournament`: Benchmark arena executing round-robin tournaments across arbitrary agent configurations with strictly balanced first-mover advantage, generating head-to-head cross-tables and leaderboards. Options: `--games <N>` (alias `--rounds`), `--board-size <N>` (alias `--size`), `--pie-rule`.
- **Reward Encoding**: Black win yields `[+1.0, -1.0]`, White win yields `[-1.0, +1.0]`.

---

### 1.3 2048 / Tzf8 (`tzf8` Crate)

Implemented in [`tzf8`](file:///home/pankaj/Projects/ai/tzf8) (with a lightweight reference version in [`mcts-envs/src/tzf8.rs`](file:///home/pankaj/Projects/ai/mcts-envs/src/tzf8.rs)).

- **Type**: Single-Player, Stochastic MDP.
- **Board Architecture**: Highly optimized 64-bit bitboard (`u64`) where each of the 16 cells occupies a 4-bit nibble encoding powers of two ($0 \implies 0, 1 \implies 2, 2 \implies 4, \dots, 11 \implies 2048$).
- **Performance**: Precomputed $65,536$-entry lookup tables (LUT) execute row shifts, tile merges, and score additions in $O(1)$ CPU operations.
- **Action Space**: 4 directional shifts: `Left`, `Right`, `Up`, `Down`.
- **Chance Mechanics & Delta-Branching**:
  - After each valid slide, a random tile spawns (value 2 with $90\%$ probability, value 4 with $10\%$ probability) in an empty cell.
  - The stochastic spawn is emitted via `StepDelta = TileSpawn { pos: u8, val: u16 }`.
  - Enables exact sample-mean Expectimax search down flat `TreeStore` arrays without dummy chance nodes.
- **Evaluators**:
  - `CornerHeuristicEvaluator`: Snake monotonicity (rewarding high tiles anchored in a corner), empty cell bonus, and edge smoothness.
  - `RolloutEvaluator`: Monte Carlo random rollouts to terminal states.
  - `UniformEvaluator`: Fast uniform prior baseline.
- **Agents**:
  - `MctsAgent`: Supports pure UCT, dynamic Min-Max normalized UCT (`mcts-norm`), and normalized PUCT with FPU.
  - `HeuristicAgent`: 1-ply greedy lookahead selecting the shift that maximizes the corner heuristic.
  - `RandomAgent`: Uniform random legal actions.
  - `HumanAgent`: Interactive arrow/WASD keyboard player.
- **CLI Utilities**:
  - `tzf8-play`: Interactive terminal game with ANSI board rendering, live MCTS move evaluation, and visit distributions.
  - `tzf8-tournament`: Benchmark arena evaluating agents across $N$ identical seeds, tracking average score, moves/second, and max tile distribution ($\ge 2048, \ge 4096, \ge 8192, \ge 16384$).

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

### 1.5 Blokus Classic & Duo (`BlokusWorld<B, P>` & `TurnBasedDynamics<BlokusWorld<B, P>>`)

Implemented in [`blokus`](file:///home/pankaj/Projects/ai/blokus).

- **Type**: Multi-Agent (2 or 4 Players), Perfect Information, Sequential Placement.
- **Board**:
  - `Blokus Classic`: $20 \times 20$ grid, 4 players (89 polyomino squares per player).
  - `Blokus Duo`: $14 \times 14$ grid, 2 players.
- **Pieces**: 21 standard polyominoes (monomino to pentominoes) normalized into 91 canonical 2D orientations.
- **Placement Rules**:
  - First move must cover starting square (corner in Classic, $(4,4)$ and $(9,9)$ in Duo).
  - Subsequent moves must touch at least one corner of the player's previously placed pieces, with strictly no edge-to-edge contact with own pieces.
- **Scoring**: $-1$ per unplaced square; $+15$ bonus for placing all 21 pieces; additional $+5$ bonus if monomino is placed last.
- **Traits Implemented**:
  - `World`: Full $P$-player match referee with simultaneous joint-action interface and rank rewards.
  - `TurnBasedWorld`: In-place single-action transition returning normalized rank reward vector $\mathbf{r} \in [-1.0, 1.0]^P$ summing to $0.0$.
  - `TurnBasedDynamics<BlokusWorld>`: Universal planning dynamics powering `MctsAgent`.

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

### Step 2: Implement `TurnBasedWorld` (or `AgentDynamics`)

For turn-based board games, implement [`World`](mcts_traits::World) and [`TurnBasedWorld`](mcts_traits::TurnBasedWorld):

```rust
use mcts_traits::{StepOutcome, TurnBasedWorld, World};

pub struct MyGameWorld;

impl World for MyGameWorld {
    type WorldState = MyGameState;
    type Action = MyAction;
    type Observation = MyGameState;

    fn n_players(&self) -> usize { 2 }
    fn initial(&self) -> Self::WorldState { MyGameState { /* ... */ } }
    fn observe(&self, ws: &Self::WorldState, _p: usize) -> Self::Observation { ws.clone() }
    fn actions(&self, ws: &Self::WorldState, p: usize, out: &mut Vec<Self::Action>) {
        if p == self.current_player(ws) {
            // populate legal actions into `out` without heap allocations
        }
    }
    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        let active = self.current_player(ws);
        let outcome = self.step_action(ws, &joint[active]);
        (outcome.reward.to_vec(), outcome.terminated)
    }
    fn terminal(&self, ws: &Self::WorldState) -> bool { /* check terminal */ false }
}

impl TurnBasedWorld for MyGameWorld {
    type StepReward = [f32; 2];

    fn current_player(&self, ws: &Self::WorldState) -> usize {
        // Return 0 or 1
        0
    }

    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> StepOutcome<Self::StepReward> {
        // Apply action in-place to `ws` and return rewards on the stack
        let reward = [0.0, 0.0];
        let terminated = false;
        StepOutcome::new(reward, terminated)
    }
}
```

### Step 3: Connect to MCTS via `TurnBasedDynamics`
Wrap your world referee with `TurnBasedDynamics` to immediately obtain zero-allocation planning dynamics:

```rust
use mcts_traits::TurnBasedDynamics;

let world = MyGameWorld;
let dynamics = TurnBasedDynamics::new(world);
```

Your custom dynamics can immediately plug into any scheduler (`SequentialScheduler`, `BatchedScheduler`, `MultiGameScheduler`) and selection/backup policy without any handwritten wrapper boilerplate!

