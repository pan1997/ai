# RFC: Asynchronous Actor-Learner Architecture (Rust Self-Play $\leftrightarrow$ Python PyTorch Learner)

This document specifies the low-level architecture, communication protocols, memory layouts, and crate boundaries for **decoupling high-throughput Rust MCTS self-play from Python neural network training**.

---

## 1. Executive Summary & Problem Formulation

Modern reinforcement learning systems for combinatorial games (such as AlphaZero, MuZero, and KataGo) exhibit fundamentally asymmetrical compute demands:
1. **Tree Search / Simulation Actors**:
   - Dominated by branch prediction, rapid pointerless tree traversal, integer bitboard operations, and fine-grained state updates.
   - Best executed in native compiled languages (Rust) using Structure-of-Arrays (SoA) memory layouts and zero-allocation hot paths.
2. **Gradient-Based Policy & Value Learner**:
   - Dominated by dense matrix multiplications, autograd graphs, cuDNN kernels, and optimizer state tracking.
   - Best executed in high-productivity tensor frameworks (PyTorch) running on dedicated GPU accelerators.

Coupling these two domains in a single process (via embedded Python interpreters like `pyo3` or embedded C++ bindings) introduces severe runtime frictions:
- **Global Interpreter Lock (GIL) Contention**: Inhibits multi-threaded MCTS scaling across CPU cores.
- **CUDA Kernel Serialization**: Individual MCTS worker threads issuing tiny batch-size-1 inference calls stall the GPU pipeline, causing execution bubbles and low Tensor Core occupancy.
- **Garbage Collection Pauses**: Python runtime GC interrupts high-frequency search actors.
- **Deployment Fragility**: Tightly couples the Rust toolchain with Python virtual environments, CUDA driver libraries, and PyTorch dynamic linkers.

### The Solution: Asynchronous Filesystem-Decoupled Actor-Learner
We partition the system into an **Asynchronous Actor-Learner Pattern**:
- **Rust Actors**: Massively parallel self-play workers generate search trajectories using MCTS, batching leaf evaluations through an in-process dynamic micro-batcher backed by ONNX Runtime (`ort`), and spooling game trajectories to disk in compact binary chunks.
- **Python Learner**: An autonomous PyTorch training daemon ingests binary chunks into an in-memory replay buffer, updates neural network weights, and periodically exports updated ONNX graphs using atomic filesystem renames.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Rust Self-Play Cluster                        │
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐                                  │
│  │ MCTS Actor 1 │ ... │ MCTS Actor N │                                  │
│  └──────┬───────┘     └──────┬───────┘                                  │
│         │ EvalRequest        │ EvalRequest                              │
│         └──────────┬─────────┘                                          │
│                    ▼                                                    │
│       ┌─────────────────────────┐                                       │
│       │   Inference Dispatcher  │                                       │
│       │ (Dynamic Micro-Batcher) │                                       │
│       └────────────┬────────────┘                                       │
│                    │ Batched NCHW Tensor (B <= max_batch_size)          │
│                    ▼                                                    │
│       ┌─────────────────────────┐                                       │
│       │   ort::Session (CUDA)   │                                       │
│       └────────────▲────────────┘                                       │
│                    │ Atomic Hot-Swap                                    │
│       ┌────────────┴────────────┐                                       │
│       │   Model Sync Watcher    │ ◄────────── latest.onnx ────────────┐ │
│       └─────────────────────────┘             (Atomic Rename)         │ │
│                    │                                                  │ │
│  ┌─────────────────┴─────────────────┐                                │ │
│  │     Trajectory Chunk Spooler      │                                │ │
│  └─────────────────┬─────────────────┘                                │ │
└────────────────────┼──────────────────────────────────────────────────┼─┘
                     │ Chunked Binary Files                             │
                     │ (traj_<worker>_<ts>.bin)                         │
                     ▼                                                  │
┌───────────────────────────────────────────────────────────────────────┼─┐
│                             Python Learner                            │ │
│                                                                       │ │
│         ┌─────────────────────────────────────┐                       │ │
│         │ Replay Buffer Ingest (np.fromfile)  │                       │ │
│         └──────────────────┬──────────────────┘                       │ │
│                            ▼                                          │ │
│         ┌─────────────────────────────────────┐                       │ │
│         │        PyTorch Training Loop        │                       │ │
│         │    (Cross-Entropy + MSE Losses)     │                       │ │
│         └──────────────────┬──────────────────┘                       │ │
│                            ▼                                          │ │
│         ┌─────────────────────────────────────┐                       │ │
│         │         Atomic ONNX Exporter        │───────────────────────┘ │
│         │      (latest.onnx.tmp -> .onnx)     │                         │
│         └─────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Workspace Crate Boundaries & Invariant Compliance

To adhere strictly to **`AGENTS.md §1 & §2`**, crate dependencies must remain rigorously decoupled:

```
mcts-traits/  --> Zero-dependency traits and interfaces (abstractions only)
mcts-engine/  --> Core Structure-of-Arrays (SoA) engine, selection, backup, and schedulers
mcts-envs/    --> Reference environments, benchmark games, and heuristic/rollout evaluators
connect4/     --> Dedicated Connect 4 game engine, MCTS agents, and interactive CLI players
blokus/       --> Dedicated Blokus (Duo & Classic) game engine, multi-agent MCTS agents
tzf8/         --> Dedicated 2048 Expectimax game engine, chance-node MCTS agents
hex/          --> Dedicated Hex game engine, DSU connectivity tracking, MCTS agents
mcts-onnx/    --> [NEW] ONNX Runtime (ort) inference client, dynamic micro-batcher,
              --> model sync watcher, and binary trajectory spooler
```

### Dependency Invariants
1. `mcts-traits` **must never** depend on `ort`, `ndarray`, `notify`, `flume`, or `mcts-engine`.
2. `mcts-engine` depends **only** on `mcts-traits` and minimal math (`rand`). It **must never** link against `ort` or C/C++ neural network runtimes.
3. The new **`mcts-onnx`** crate links against:
   - `mcts-traits` (implements `Model<S>` and `BatchedModel<S>`)
   - `mcts-engine` (uses `TreeStore` and `MatchDriver` for trajectory collection)
   - `ort` (ONNX Runtime C bindings with CUDA / TensorRT / CPU execution providers)
   - `ndarray` (contiguous tensor views and layout manipulation)
   - `flume` (lockless MPSC / rendezvous channels for batch queuing)
   - `notify` (cross-platform filesystem inotify/kqueue event watching)
4. Environment crates (`connect4`, `hex`, etc.) remain zero-dependency relative to `mcts-onnx`, exposing canonical tensor conversion methods via generic traits.

---

## 3. ONNX Graph Partitioning: AlphaZero vs. MuZero

To support dynamic tree search batching without runtime shape errors, models must be exported with **dynamic batch axes**.

### 3.1 AlphaZero: Single Dual-Headed Network
In AlphaZero, the model directly evaluates full observation states:

$$f_\theta(s) \to (\mathbf{p}, \mathbf{v})$$

- $\mathbf{p} \in \Delta^{|\mathcal{A}|}$: Policy prior vector over legal actions.
- $\mathbf{v} \in [-1, +1]^N$: Value estimate vector for all $N$ players.

#### ONNX Signature
- **Input Names**: `["state"]`
  - Shape: `(batch_size, channels, height, width)` (float32)
- **Output Names**: `["policy", "value"]`
  - Policy Shape: `(batch_size, num_actions)` (float32 logits or softmax probabilities)
  - Value Shape: `(batch_size, num_players)` (float32, tanh activation)
- **Dynamic Axes**:
  ```python
  dynamic_axes = {
      "state": {0: "batch_size"},
      "policy": {0: "batch_size"},
      "value": {0: "batch_size"},
  }
  ```

---

### 3.2 MuZero: Split-Session Lifecycle
In MuZero, the game dynamics are learned in a latent embedding space $\mathcal{S}_{\text{latent}} \subset \mathbb{R}^D$.

> [!CAUTION]
> **Do not attempt to export recurrent unrolled loops into a single monolithic ONNX graph.**
> MCTS tree traversal depth varies dynamically per simulation. Compiling a fixed-depth unrolled loop prevents MCTS from branching arbitrarily.

Instead, MuZero is partitioned into **two distinct ONNX sessions** that align directly with the MCTS node lifecycle:

```
[Observation o] ──► initial.onnx ──► Latent Root s_0, Policy p_0, Value v_0
                         │
                    Action a_1
                         ▼
[s_0, a_1]      ──► recurrent.onnx ─► Latent Next s_1, Reward r_1, Policy p_1, Value v_1
                         │
                    Action a_2
                         ▼
[s_1, a_2]      ──► recurrent.onnx ─► Latent Next s_2, Reward r_2, Policy p_2, Value v_2
```

#### Session 1: `initial.onnx` (Representation $h_\theta$ + Prediction $f_\theta$)
Executed **once per search** at the root node.
- **Mathematical Function**:
  $$o \to (s_0, \mathbf{p}_0, \mathbf{v}_0)$$
- **Inputs**:
  - `observation`: `(batch_size, channels, height, width)`
- **Outputs**:
  - `latent_state`: `(batch_size, latent_dim)`
  - `policy`: `(batch_size, num_actions)`
  - `value`: `(batch_size, num_players)`
- **Invocation**: Pre-seeds the root node in `TreeStore`.

#### Session 2: `recurrent.onnx` (Dynamics $g_\theta$ + Prediction $f_\theta$)
Executed **on every edge expansion** during tree descent.
- **Mathematical Function**:
  $$(s_{k-1}, a_k) \to (s_k, \mathbf{r}_k, \mathbf{p}_k, \mathbf{v}_k)$$
- **Inputs**:
  - `latent_state`: `(batch_size, latent_dim)`
  - `action`: `(batch_size,)` (int64) or one-hot `(batch_size, num_actions)`
- **Outputs**:
  - `next_latent_state`: `(batch_size, latent_dim)`
  - `reward`: `(batch_size, num_players)`
  - `policy`: `(batch_size, num_actions)`
  - `value`: `(batch_size, num_players)`
- **Invocation**: Evaluates and expands leaves during `BatchedScheduler` or `SequentialScheduler` descent.

#### Natural Alignment with `mcts-traits`:
In `mcts-traits`, `AgentDynamics` decouples external reality from internal planning:
- `AgentDynamics::State = [f32; LATENT_DIM]` (the latent vector).
- `AgentDynamics::step(s, a)` queries `recurrent.onnx`, returning the next latent state and intermediate reward $r$.
- `Model::evaluate(s)` returns the predicted policy priors $\mathbf{p}$ and value estimates $\mathbf{v}$.

---

## 4. Rust Inference Dispatcher & Dynamic Micro-Batching

### 4.1 The Kernel Serialization Problem
In `ort 2.x`, executing `session.run()` requires mutable or synchronized access to execution buffers. If 32 search worker threads hold independent references or serialize through a single mutex to run batch-size-1 inference:
1. Each GPU kernel launch incurs driver latency ($\approx 10\text{–}30\,\mu\text{s}$).
2. CUDA stream queues serialize.
3. GPU Tensor Cores sit idle >80% of the time.

### 4.2 Dynamic Micro-Batching Architecture
The inference engine uses a **single dedicated inference thread** that owns the `ort::Session` and continuously drains an asynchronous lockless queue (`flume`):

```rust
pub struct EvalRequest {
    /// Flattened observation features (Channels * Height * Width)
    pub features: Vec<f32>,
    /// One-shot response channel back to the calling search thread
    pub respond_to: tokio::sync::oneshot::Sender<Evaluation>,
}
```

#### Dispatcher Loop Algorithm:
1. **Wait for First Item**: Blocks until at least 1 request arrives in `req_rx`.
2. **Dynamic Accumulation**: Collects additional requests until:
   - `queue.len() == max_batch_size` (e.g., 64 or 128), OR
   - The deadline $t_{\text{now}} + \Delta t_{\text{max}}$ expires (e.g., $\Delta t_{\text{max}} = 2.0\,\text{ms}$).
3. **Contiguous Tensor Stacking**: Flattens all queued features into a preallocated contiguous `ndarray::Array4<f32>` buffer without intermediate vector allocations.
4. **Hardware Forward Pass**: Runs `session.run()` once for the combined batch.
5. **Fan-Out Dispatch**: Slices the output tensor along `Axis(0)` and sends `Evaluation` structs back across each request's `respond_to` channel.

```rust
// mcts-onnx/src/inference.rs
pub fn run_inference_loop(
    mut session: ort::session::Session,
    req_rx: flume::Receiver<EvalRequest>,
    reload_rx: flume::Receiver<PathBuf>,
    config: BatcherConfig,
) {
    let mut batch_features = Vec::with_capacity(config.max_batch_size * config.feature_size);
    let mut request_batch = Vec::with_capacity(config.max_batch_size);

    while let Ok(first_req) = req_rx.recv() {
        // 1. Check for atomic weight reload requests
        if let Ok(new_model_path) = reload_rx.try_recv() {
            if let Ok(new_session) = create_session(&new_model_path, &config) {
                session = new_session;
                println!("[Inference] Hot-swapped ONNX weights to {:?}", new_model_path);
            }
        }

        // 2. Micro-batch accumulation up to timeout
        request_batch.clear();
        request_batch.push(first_req);
        let deadline = Instant::now() + config.max_latency;

        while request_batch.len() < config.max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            if let Ok(req) = req_rx.recv_timeout(remaining) {
                request_batch.push(req);
            } else {
                break;
            }
        }

        // 3. Assemble contiguous NCHW tensor
        batch_features.clear();
        for req in &request_batch {
            batch_features.extend_from_slice(&req.features);
        }

        let current_batch_size = request_batch.len();
        let input_tensor = ndarray::Array4::from_shape_vec(
            (current_batch_size, config.channels, config.height, config.width),
            batch_features.clone(),
        ).expect("Shape mismatch");

        // 4. Run batched ONNX session
        let outputs = session
            .run(ort::inputs![input_tensor.view()].expect("Tensor view failed"))
            .expect("Inference execution failed");

        let policy_tensor = outputs[0].extract_tensor::<f32>().expect("Policy output");
        let value_tensor = outputs[1].extract_tensor::<f32>().expect("Value output");

        // 5. Fan-out responses
        let policy_view = policy_tensor.view();
        let value_view = value_tensor.view();

        for (i, req) in request_batch.drain(..).enumerate() {
            let priors = policy_view.index_axis(ndarray::Axis(0), i).to_vec();
            let values = value_view.index_axis(ndarray::Axis(0), i).to_vec();
            let _ = req.respond_to.send(Evaluation { priors, values });
        }
    }
}
```

---

### 4.3 Implementing `Model<S>` and `BatchedModel<S>`

The search threads communicate with the inference thread through an `OnnxModelClient` implementing the traits from `mcts-traits`:

```rust
pub struct OnnxModelClient<S, E> {
    req_tx: flume::Sender<EvalRequest>,
    encoder: E,
    _marker: std::marker::PhantomData<S>,
}

impl<S, E: StateEncoder<S>> Model<S> for OnnxModelClient<S, E> {
    fn evaluate(&self, s: &S) -> Evaluation {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut features = Vec::with_capacity(E::CHANNELS * E::HEIGHT * E::WIDTH);
        self.encoder.encode(s, &mut features);

        let _ = self.req_tx.send(EvalRequest {
            features,
            respond_to: resp_tx,
        });

        // Block this search actor thread until the batched GPU forward pass finishes
        resp_rx.blocking_recv().expect("Inference worker dropped channel")
    }
}
```

> [!TIP]
> **Zero Allocation with Reusable Channels**:
> For ultra-high-throughput setups, each worker thread can maintain a dedicated reusable rendezvous channel or fixed memory slot, avoiding per-request `oneshot::channel` allocation.

---

## 5. Atomic Model Hot-Swapping & Filesystem Synchronization

To allow training in Python while Rust continues self-play uninterrupted:

### 5.1 The Atomic Publish Protocol
Writing directly to `model.onnx` causes race conditions where Rust attempts to read a half-written file.
1. Python writes weights to a temporary file: `models/model_step_1000.onnx.tmp`.
2. Python executes `os.replace("models/model_step_1000.onnx.tmp", "models/latest.onnx")`.
3. On POSIX and Windows filesystems, `rename()` / `os.replace()` is an **atomic directory entry replacement**. The inode is swapped instantaneously.

### 5.2 Filesystem Watcher (`notify` Crate)
Rust spawns a background filesystem watcher monitoring the `models/` directory:

```rust
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

pub fn start_model_watcher(
    model_path: PathBuf,
    reload_tx: flume::Sender<PathBuf>,
) -> RecommendedWatcher {
    let watch_dir = model_path.parent().expect("Invalid model parent directory").to_path_buf();
    let target_filename = model_path.file_name().expect("Invalid filename").to_os_string();

    let mut watcher = RecommendedWatcher::new(
        move |res: Result<Event, _>| {
            if let Ok(event) = res {
                if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                    for path in &event.paths {
                        if path.file_name() == Some(&target_filename) {
                            // Debounce and verify non-zero file size before triggering reload
                            if let Ok(metadata) = std::fs::metadata(path) {
                                if metadata.len() > 0 {
                                    let _ = reload_tx.send(path.clone());
                                }
                            }
                        }
                    }
                }
            }
        },
        Config::default(),
    ).expect("Failed to initialize file watcher");

    watcher.watch(&watch_dir, RecursiveMode::NonRecursive).expect("Failed to watch directory");
    watcher
}
```

---

## 6. Binary Trajectory Spooling Wire Specification

Instead of text (JSON, CSV) or high-overhead schema encodings (Protobuf, FlatBuffers), trajectories are written in a **packed, fixed-stride binary format**.

### 6.1 Trajectory File Structure
Each chunk file has a 64-byte aligned header followed by $N$ contiguous step records:

```
┌──────────────────────────────────────────────────────────────┐
│                    HEADER (64 Bytes)                         │
├─────────────────┬──────────────┬──────────────┬──────────────┤
│ magic: [u8; 4]  │ version: u32 │ game_id: u32 │ num_steps:u32│
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ obs_dtype: u32  │ channels: u32│ height: u32  │ width: u32   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ action_dim: u32 │ num_players:u32│stride_bytes:u64│ RESERVED │
├─────────────────┴──────────────┴──────────────┴──────────────┤
│                                                              │
│                    CONTIGUOUS STEP RECORDS                   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ Step 0:                                                      │
│   observation:      [f32; C * H * W]                         │
│   action_mask:      [u8; (action_dim + 7) / 8]               │
│   policy_target:    [f32; action_dim]   (from MCTS visit %)  │
│   value_target:     [f32; num_players]  (from GameOutcome)   │
│   action_taken:     u32                                      │
│   reward_target:    [f32; num_players]  (for MuZero)         │
├──────────────────────────────────────────────────────────────┤
│ Step 1: ...                                                  │
└──────────────────────────────────────────────────────────────┘
```

#### Binary Layout Details
- **`magic`**: `b"MCTS"` (`[0x4D, 0x43, 0x54, 0x53]`)
- **`version`**: `1`
- **`obs_dtype`**: `0` = float32, `1` = bitpacked_u64
- **Endianness**: Standard Little-Endian (`<`) for all integers and floats.
- **Fixed Stride**: Because every step record is identical in byte size:
  $$\text{RecordStride} = (C \cdot H \cdot W \cdot 4) + \lceil|\mathcal{A}| / 8\rceil + (|\mathcal{A}| \cdot 4) + (N_{\text{players}} \cdot 4) + 4 + (N_{\text{players}} \cdot 4)$$
  Python can slice directly into any arbitrary step $k$ at byte offset:
  $$\text{Offset}(k) = 64 + k \cdot \text{RecordStride}$$

---

### 6.2 Atomic Flush Protocol (Rust)
1. Each Rust worker buffers steps into local memory.
2. When the buffer reaches `CHUNK_SIZE` steps (e.g. 1,000 steps $\approx 1\text{–}3\,\text{MB}$):
   - Opens `spool/traj_<worker_id>_<timestamp>.bin.tmp`.
   - Writes header and records using a single `BufWriter`.
   - Calls `file.flush()` and `file.sync_all()`.
   - Renames file to `spool/traj_<worker_id>_<timestamp>.bin`.
3. The file appears atomically in the directory, preventing Python from encountering incomplete payloads.

---

### 6.3 Zero-Copy Ingestion in Python
Python's `trainer.py` scans `spool/` for `traj_*.bin`, reads chunks directly via `numpy.fromfile()` or `np.memmap()`, pushes them into the replay buffer, and deletes the processed chunk:

```python
# python/learner/spool_reader.py
import struct
import numpy as np
from pathlib import Path

HEADER_FORMAT = "<4sIIIIIIIIQQ"  # 64 bytes
HEADER_SIZE = 64

def read_trajectory_chunk(chunk_path: Path):
    with open(chunk_path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        magic, version, game_id, num_steps, obs_dtype, c, h, w, action_dim, num_players, stride = struct.unpack(
            HEADER_FORMAT, header_bytes
        )
        assert magic == b"MCTS", f"Invalid magic: {magic}"

        # Read remaining records into flat numpy array
        buffer = f.read()

    obs_len = c * h * w
    policy_len = action_dim
    val_len = num_players

    # Fast struct decomposition with numpy slicing
    dt = np.dtype([
        ("obs", np.float32, (obs_len,)),
        ("mask", np.uint8, ((action_dim + 7) // 8,)),
        ("policy", np.float32, (policy_len,)),
        ("value", np.float32, (val_len,)),
        ("action", np.uint32),
        ("reward", np.float32, (val_len,)),
    ])

    records = np.frombuffer(buffer, dtype=dt, count=num_steps)
    return records
```

---

## 7. Python Trainer Architecture (`trainer.py`)

### 7.1 Multi-Task Loss Functions

#### AlphaZero Loss:
$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( \|\mathbf{v}_\theta(s_i) - \mathbf{z}_i\|_2^2 - \boldsymbol{\pi}_i^\top \log \mathbf{p}_\theta(s_i) \right) + c \|\theta\|_2^2$$

- $\mathbf{z}_i \in [-1, +1]^N$: True game outcome return vector from `MatchDriver`.
- $\boldsymbol{\pi}_i \in \Delta^{|\mathcal{A}|}$: Normalized MCTS root visit distribution:
  $$\pi(a) = \frac{N(\text{root}, a)^{1/\tau}}{\sum_{b} N(\text{root}, b)^{1/\tau}}$$
- $\mathbf{p}_\theta(s_i)$: Model policy logits normalized via log-softmax over legal actions.

#### MuZero Loss:
In addition to policy and value, MuZero incorporates intermediate reward prediction across $K$ unrolled simulation steps:
$$\mathcal{L}(\theta) = \sum_{k=0}^K \left[ \ell_v(\mathbf{v}_k, z) + \ell_r(\mathbf{r}_k, u_k) + \ell_p(\mathbf{p}_k, \pi_k) \right] + c \|\theta\|_2^2$$

---

### 7.2 Training Loop Implementation
```python
# python/learner/trainer.py
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from spool_reader import read_trajectory_chunk

def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    spool_dir: str,
    export_path: str,
    input_shape: tuple,
    batch_size: int = 256,
    min_buffer_size: int = 10_000,
    export_interval: int = 200,
):
    spool_path = Path(spool_dir)
    replay_buffer = []
    step = 0

    print(f"[Trainer] Monitoring {spool_path} for trajectory chunks...")

    while True:
        # 1. Ingest available chunks from Rust
        for chunk in list(spool_path.glob("traj_*.bin")):
            try:
                records = read_trajectory_chunk(chunk)
                replay_buffer.extend(records)
                chunk.unlink()  # Delete immediately after memory ingest
            except (IOError, PermissionError):
                continue

        # Keep buffer bounded (FIFO sliding window)
        if len(replay_buffer) > 200_000:
            replay_buffer = replay_buffer[-200_000:]

        if len(replay_buffer) < min_buffer_size:
            time.sleep(0.2)
            continue

        # 2. Sample uniform mini-batch
        indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]

        obs = torch.tensor(np.stack([b["obs"] for b in batch]), dtype=torch.float32).cuda()
        target_policy = torch.tensor(np.stack([b["policy"] for b in batch]), dtype=torch.float32).cuda()
        target_value = torch.tensor(np.stack([b["value"] for b in batch]), dtype=torch.float32).cuda()

        # 3. Compute loss and optimize
        optimizer.zero_grad()
        pred_policy_logits, pred_value = model(obs.view(-1, *input_shape))

        loss_policy = -torch.sum(target_policy * torch.log_softmax(pred_policy_logits, dim=-1)) / batch_size
        loss_value = torch.mean((pred_value - target_value) ** 2)
        total_loss = loss_policy + loss_value

        total_loss.backward()
        optimizer.step()
        step += 1

        # 4. Periodic atomic export
        if step % export_interval == 0:
            export_alphazero_onnx(model, export_path, input_shape)
            print(f"[Trainer] Step {step} | Loss: {total_loss.item():.4f} | Exported: {export_path}")
```

---

## 8. Game Observation Tensor Representations

To preserve the zero-dependency property of `mcts-traits` and `mcts-engine`, environments implement a generic `TensorRepresentable` trait:

```rust
// mcts-traits/src/model.rs
pub trait TensorRepresentable {
    const CHANNELS: usize;
    const HEIGHT: usize;
    const WIDTH: usize;
    fn encode_tensor(&self, out: &mut [f32]);
}
```

### 8.1 Connect 4 (`Connect4State<R, C>`)
- **Dimensions**: `(2, R, C)`
  - Channel 0: $1.0$ where Red checker is present, else $0.0$.
  - Channel 1: $1.0$ where Yellow checker is present, else $0.0$.
- **Action Space**: Column indices $0 \dots C-1$ ($|\mathcal{A}| = C$).

### 8.2 Hex (`HexState<N>`)
- **Dimensions**: `(2, N, N)`
  - Channel 0: $1.0$ where Black stone is placed.
  - Channel 1: $1.0$ where White stone is placed.
- **Action Space**: Cell indices $0 \dots N^2-1$, plus optional pie-rule swap action ($|\mathcal{A}| = N^2$ or $N^2 + 1$).

### 8.3 2048 (`Tzf8State`)
- **Dimensions**: `(16, 4, 4)`
  - One-hot binary planes for each tile value power: $2^1, 2^2, \dots, 2^{16}$.
- **Action Space**: 4 directional slides: Left, Right, Up, Down ($|\mathcal{A}| = 4$).

### 8.4 Blokus Classic & Duo (`BlokusState<B, P>`)
- **Dimensions**: `(P + 1, B, B)`
  - Channels $0 \dots P-1$: Stones placed by Player $p$.
  - Channel $P$: Valid legal corner placement indicator mask.
- Additional auxiliary input: `(P, 21)` binary vector indicating remaining unplayed polyomino pieces.

---

## 9. Performance Budget & Hardware Sizing

| Metric | Target (Single RTX 4090 + 32-core CPU) | Sizing Rationale |
|---|---|---|
| **MCTS Worker Threads** | 24–28 threads | Leaves 4 cores for OS, inference dispatch, and Python trainer |
| **Inference Batch Size** | 64 – 128 | Saturates GPU Tensor Cores without exceeding latency budget |
| **Max Batch Latency** | 2.0 ms | Prevents MCTS workers from waiting excessively on small batches |
| **Inference Throughput** | >50,000 leaf evals / sec | Enables $\approx 500$ 100-sim MCTS decisions / sec across all games |
| **Trajectory Chunk Size** | 1,000 steps ($\approx 1.5\text{–}3.0\,\text{MB}$) | Minimizes filesystem inode churn while ensuring low trainer lag |
| **Python Buffer Ingestion** | <5 ms per chunk | Instantaneous via `numpy.frombuffer` |
| **Model Reload Latency** | <150 ms | Compiles ONNX session in background; atomic pointer swap is $O(1)$ |

---

## 10. Phased Implementation Roadmap

1. **Phase 1: Architecture Specification & Low-Level Design** *(Completed in this RFC)*:
   - Finalize crate boundaries, tensor encoding traits, wire protocol, and ONNX graph signatures.
2. **Phase 2: Trajectory Spooling & Python Replay Buffer**:
   - Implement `mcts-onnx::spool` writer and `python/learner/spool_reader.py`.
   - Verify binary round-trip serialization and zero-copy numpy decoding.
3. **Phase 3: `mcts-onnx` Dynamic Micro-Batcher & Session Hot-Swap**:
   - Implement `InferenceDispatcher`, `OnnxModelClient`, and `start_model_watcher`.
   - Validate hot-swapping under continuous search load without panics or memory leaks.
4. **Phase 4: End-to-End AlphaZero Verification on Connect 4 & Hex**:
   - Train self-play agent from scratch against uniform random baseline.
   - Benchmark throughput curves (evals/sec, moves/sec, GPU utilization).
5. **Phase 5: MuZero 2-Session Partitioning**:
   - Implement `initial.onnx` and `recurrent.onnx` bindings integrated with `AgentDynamics`.
