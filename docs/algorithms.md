# Algorithms & Mathematical Formulations

This document provides the exact mathematical formulations, pseudocode, and algorithmic behaviors implemented across `mcts-engine`.

---

## 1. Selection Policies

During the selection phase, the search traverses from the root down to a leaf node by iteratively selecting the child edge that maximizes an acquisition function.

### 1.1 Classic Upper Confidence Bounds for Trees (UCT)

Implemented in [`UctSelection`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/uct.rs).

For a node $s$ with active agent $i$ and child edge $a$:

$$\text{Score}(s, a) = \begin{cases} +\infty & \text{if } N(s, a) = 0 \\ Q_i(s, a) + c_{\text{uct}} \sqrt{\frac{\ln(N(s) + 1)}{N(s, a)}} & \text{if } N(s, a) > 0 \end{cases}$$

Where:
- $N(s, a)$ is the visit count of edge $a$.
- $N(s) = \sum_{a'} N(s, a')$ is the total visits of the parent node $s$.
- $Q_i(s, a)$ is the running mean value estimate for active player $i$.
- $c_{\text{uct}}$ is the exploration constant (defaults to $\sqrt{2} \approx 1.4142$).

Unvisited edges receive infinite priority to ensure every legal action is sampled at least once before exploitation begins.

---

### 1.2 Predictor Upper Confidence Bounds for Trees (PUCT) with Virtual Loss

Implemented in [`MultiAgentPuctSelection`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/puct.rs).

In modern deep RL (AlphaZero, Silver et al., 2017), search is guided by a prior policy probability distribution $P(s, a)$ output by a neural network:

$$\text{Score}(s, a) = Q_{\text{eff}}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N_{\text{eff}}(s, a)}$$

#### Virtual Loss Adjustment
To enable batched or parallel simulation without selecting the exact same path repeatedly before evaluations return, a virtual loss penalty $v_{\text{loss}} \ge 0$ is temporarily applied to traversed edges:

$$N_{\text{eff}}(s, a) = N(s, a) + v_{\text{loss}}$$

$$Q_{\text{eff}}(s, a) = \begin{cases} \frac{Q(s, a) \cdot N(s, a) - v_{\text{loss}}}{N_{\text{eff}}(s, a)} & \text{if } N_{\text{eff}}(s, a) > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

This temporarily depresses the edge's estimated value and increases its effective visit count, forcing subsequent parallel searches within the same batch to explore alternative branches. When the evaluation returns, the virtual loss is removed and true statistics are updated.

---

### 1.3 Gumbel AlphaZero Selection

Implemented in [`GumbelPuctSelection`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/gumbel.rs).

Based on *Policy Improvement by Planning with Gumbel* (Danihelka et al., 2022). Standard PUCT at small simulation budgets can fail to visit optimal moves due to prior bias. Gumbel AlphaZero fixes this by injecting Gumbel noise into the prior policy logits **at the root node**:

$$g(a) \sim \text{Gumbel}(0, 1) = -\ln(-\ln(U)), \quad U \sim \text{Uniform}(0, 1)$$

$$\text{Score}_{\text{root}}(s, a) = Q_{\text{eff}}(s, a) + c_{\text{puct}} \cdot \frac{\sqrt{N(s)}}{1 + N_{\text{eff}}(s, a)} \cdot \big(\ln P(s, a) + g(a)\big)$$

#### Interior Determinism
At non-root / interior nodes, standard deterministic PUCT is preserved:

$$\text{Score}_{\text{interior}}(s, a) = Q_{\text{eff}}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N_{\text{eff}}(s, a)}$$

This guarantees consistent policy improvement without accumulating stochastic variance down the tree.

---

### 1.4 Dynamic Min-Max Normalized UCT

Implemented in [`NormalizedUctSelection`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/normalized.rs).

#### Motivation: The Scale Mismatch Problem
Standard UCT assumes $Q \in [0, 1]$ or reward scales comparable with the exploration bonus $c_{\text{uct}} \sqrt{\frac{\ln N}{n}} \approx 1.4$. In domains with unbounded or large score metrics (e.g. 2048 where scores reach $30,000+$, or Blokus with negative scores):
- If $Q \gg c_{\text{uct}}$, the exploitation term dwarfs exploration, collapsing MCTS into a greedy rollout and pruning promising branches prematurely.
- If $Q \ll c_{\text{uct}}$, search becomes unguided random exploration.

#### Mathematical Formulation
At node $s$, compute dynamic sibling bounds across all visited children $\mathcal{C}(s)$:

$$Q_{\min}(s) = \min_{a' \in \mathcal{C}(s)} Q(s, a'), \quad Q_{\max}(s) = \max_{a' \in \mathcal{C}(s)} Q(s, a')$$

Rescale $Q(s, a)$ dynamically to $[0, 1]$:

$$Q_{\text{norm}}(s, a) = \begin{cases} \frac{Q(s, a) - Q_{\min}(s)}{Q_{\max}(s) - Q_{\min}(s)} & \text{if } Q_{\max}(s) > Q_{\min}(s) \\ 0.5 & \text{if } Q_{\max}(s) = Q_{\min}(s) \end{cases}$$

The selection score balances normalized exploitation with exploration:

$$\text{Score}(s, a) = \begin{cases} +\infty & \text{if } N(s, a) = 0 \\ Q_{\text{norm}}(s, a) + c_{\text{uct}} \sqrt{\frac{\ln(N(s) + 1)}{N(s, a)}} & \text{if } N(s, a) > 0 \end{cases}$$

---

### 1.5 Dynamic Min-Max Normalized PUCT (MuZero-Style)

Implemented in [`NormalizedPuctSelection`](file:///home/pankaj/Projects/ai/mcts-engine/src/selection/normalized.rs).

Adopting the normalization framework established in MuZero (Schrittwieser et al., 2020), $Q$-values are bounded using sibling min-max statistics, combined with a First Play Urgency (FPU) value for unvisited actions:

$$Q_{\text{norm}}(s, a) = \begin{cases} \frac{Q_{\text{eff}}(s, a) - Q_{\min}(s)}{Q_{\max}(s) - Q_{\min}(s)} & \text{if } N_{\text{eff}}(s, a) > 0 \text{ and } Q_{\max}(s) > Q_{\min}(s) \\ 0.5 & \text{if } N_{\text{eff}}(s, a) > 0 \text{ and } Q_{\max}(s) = Q_{\min}(s) \\ Q_{\min}(s) \text{ (or } 0.0) & \text{if } N_{\text{eff}}(s, a) = 0 \end{cases}$$

$$\text{Score}(s, a) = Q_{\text{norm}}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N_{\text{eff}}(s, a)}$$

This guarantees that policy priors $P(s, a) \in [0, 1]$ properly weigh against value estimates regardless of underlying game score units.

---

## 2. Backup Strategies

Once a leaf node is evaluated (via rollout, heuristic, or neural network), the return must be backpropagated up the traversed trajectory $\tau = \{(s_0, a_0), (s_1, a_1), \dots, (s_{T-1}, a_{T-1})\}$.

### 2.1 Single-Agent Discounted Backup

Implemented in [`SingleAgentBackup`](file:///home/pankaj/Projects/ai/mcts-engine/src/backup/single.rs).

For MDPs and single-player environments (like 2048) with discount factor $\gamma \in [0, 1]$:

$$G_T = V(s_T)$$
$$G_t = r(s_t, a_t) + \gamma G_{t+1}, \quad \text{for } t = T-1, \dots, 0$$

The running mean $Q(s_t, a_t)$ is updated via incremental averaging:

$$N(s_t, a_t) \leftarrow N(s_t, a_t) + 1$$
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{G_t - Q(s_t, a_t)}{N(s_t, a_t)}$$

---

### 2.2 Multi-Agent Vector Backup

Implemented in [`VectorBackup`](file:///home/pankaj/Projects/ai/mcts-engine/src/backup/vector.rs).

Traditional 2-player MCTS libraries use a scalar $Q$ and negate it at each level ($Q \leftarrow -Q$). This convention is brittle and fails in:
- Multi-player games ($N > 2$).
- Cooperative or general-sum games.
- Games where consecutive actions are made by the same agent (e.g. bonus turns).

In `VectorBackup<const N: usize>`, returns are represented explicitly as vectors $\mathbf{G} \in \mathbb{R}^N$:

$$\mathbf{G}_T = [V_0(s_T), V_1(s_T), \dots, V_{N-1}(s_T)]^\top$$
$$\mathbf{G}_t = \mathbf{r}(s_t, a_t) + \gamma \mathbf{G}_{t+1}$$

Where $\mathbf{r}(s_t, a_t) \in \mathbb{R}^N$ is the multi-agent reward vector for transition $(s_t, a_t)$.

Each edge stores an $N$-dimensional mean vector $\mathbf{Q}(s_t, a_t) \in \mathbb{R}^N$, updated element-wise:

$$\mathbf{Q}(s_t, a_t) \leftarrow \mathbf{Q}(s_t, a_t) + \frac{\mathbf{G}_t - \mathbf{Q}(s_t, a_t)}{N(s_t, a_t)}$$

During selection at node $s$, the selection policy queries $\mathbf{Q}(s, a)[i]$, where $i = \text{agent}(s)$ is the acting player at that node.

---

### 2.3 Expectimax Sample-Mean Convergence in Stochastic Environments

In stochastic Markov Decision Processes (MDPs) such as 2048 (where a random tile $\delta \sim \mathbb{P}(\cdot \mid s, a)$ spawns after each move), classical Expectimax explicitly sums over all possible chance outcomes weighted by their exact transition probabilities:

$$Q^*(s, a) = r(s, a) + \gamma \sum_{\delta \in \Omega(s, a)} \mathbb{P}(\delta \mid s, a) V^*(s', \delta)$$

In `mcts-engine`, rather than building explicit "chance nodes" with hardcoded probability tables, simulations naturally sample transitions $\delta \sim \mathbb{P}(\cdot \mid s, a)$ along edge $a$ using the environment's true generator. 

By the **Strong Law of Large Numbers (SLLN)**, as visit counts $N(s, a) \to \infty$, the empirical sample mean accumulated by [`SingleAgentBackup`](file:///home/pankaj/Projects/ai/mcts-engine/src/backup/single.rs):

$$Q_N(s, a) = \frac{1}{N(s, a)} \sum_{k=1}^{N(s, a)} G_k$$

converges almost surely to the true conditional expectation:

$$Q_N(s, a) \xrightarrow{\text{a.s.}} \mathbb{E}_{\delta \sim \mathbb{P}}[r(s, a) + \gamma V(s', \delta)] = Q^*(s, a)$$

This provides mathematically exact Expectimax planning without requiring:
1. Hardcoded chance probability distributions in the search tree.
2. Separate chance-node type hierarchies.
3. Heap allocations for outcome branching.

---

## 3. Schedulers

Schedulers orchestrate the tree search cycle across one or multiple trees.

### 3.1 `SequentialScheduler`
Executes single-threaded iterations:
1. Traverse down the tree using `SelectionPolicy` until reaching an unexpanded or terminal node.
2. Step dynamics forward along the path.
3. If unexpanded, expand the node with legal actions and evaluate with `Model`.
4. Backpropagate values using `BackupPolicy`.

### 3.2 `BatchedScheduler`
Maximizes neural network utilization on GPUs by batching evaluations within a single tree:
1. Simulates $B$ parallel trajectories from the root using virtual loss to enforce trajectory diversity.
2. Collects all unique leaf states and deduplicates them:
   $$\{s_{\text{leaf}}^{(1)}, \dots, s_{\text{leaf}}^{(B)}\} \xrightarrow{\text{dedup}} \{u_1, \dots, u_K\}, \quad K \le B$$
3. Issues a single batched neural evaluation `model.evaluate_batch(&[u_1, ..., u_K])`.
4. Expands new nodes and backpropagates returns along all $B$ paths.
5. Reverts the virtual loss on all traversed edges.

### 3.3 `MultiGameScheduler`
Executes concurrent MCTS searches across $M$ disjoint trees simultaneously:
- Ideal for asynchronous self-play data generation.
- Steps environment dynamics across active trees in parallel (`BatchedAgentDynamics::step_batch`).
- Feeds all pending leaf evaluations into a shared model batch (`BatchedModel::evaluate_batch`).
- Completely eliminates CPU-GPU synchronization stalls.

---

## 4. Tournament Arena & Statistical Benchmarking

Implemented in [`mcts_engine::arena`](file:///home/pankaj/Projects/ai/mcts-engine/src/arena.rs).

To rigorously measure agent performance and verify algorithmic improvements without hand-crafted testing scripts, the engine provides standardized tournament infrastructure.

### 4.1 Head-to-Head Matrices & Seat-Fairness Accounting

In asymmetric 2-player games (like Connect 4 or Hex, where Player 1 enjoys first-mover advantage) and 4-player games (like Blokus Classic), seating order induces strong win-rate bias. The arena mitigates this via:
- **Balanced Paired Matches**: Alternating seat assignments $(A, B)$ and $(B, A)$ across identical board seeds.
- **Uniform Seating Permutations**: Randomizing seat permutations across matches while tracking per-seat win rates.
- **Head-to-Head Cross-Table ([`H2HMatrix`](file:///home/pankaj/Projects/ai/mcts-engine/src/arena.rs))**: Tracks pairwise wins, losses, draws, and net score differentials for all $(i, j)$ agent pairs.

### 4.2 Standardized Reporting

The arena formats ANSI terminal leaderboards reporting:
- Win rate percentage ($W\%$) and total / solo / tied wins.
- Average game score and placed tiles/pieces.
- High-tier milestone distributions (e.g. tile rates $\ge 2048, \ge 4096, \ge 8192, \ge 16384$ in 2048).
- Duration and move decision throughput (moves/second).

