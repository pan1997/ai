# Task: Implement Asynchronous Actor-Learner Training Pipeline (Rust Self-Play $\leftrightarrow$ PyTorch Learner)

**Issue / Task ID**: `TASK-ACTOR-LEARNER-001`  
**Status**: Open  
**Component**: `mcts-traits`, `mcts-engine`, `mcts-onnx` (new crate), `environments/connect4`, `python/learner`  
**Tags**: `reinforcement-learning`, `alphazero`, `onnx`, `dynamic-batching`, `gpu-training`  
**Parent RFC**: [`docs/actor_learner_architecture.md`](file:///home/pankaj/Projects/ai/docs/actor_learner_architecture.md)

---

## 1. Executive Summary & Goals

This task implements an asynchronous, filesystem-decoupled **AlphaZero / MuZero Actor-Learner Pipeline** connecting Rust high-throughput MCTS self-play workers with a Python PyTorch neural network training daemon:

1. **Rust Actors (`mcts-onnx`)**: Multi-threaded self-play search workers running MCTS in Rust, querying GPU tensor inference through a lockless dynamic micro-batcher backed by ONNX Runtime (`ort`), and spooling game trajectories to disk in a compact, fixed-stride binary wire format.
2. **Python Learner (`python/learner`)**: An autonomous PyTorch training service that ingests binary trajectory chunks into a replay buffer via zero-copy memory mapping (`numpy.frombuffer`), optimizes dual-headed policy/value losses, and exports updated ONNX weights using atomic file replacement (`os.replace`).
3. **Model Hot-Swapper**: Background filesystem watcher (`notify`) detecting updated model weights and hot-swapping `ort::Session` instances in $O(1)$ time without interrupting active search threads.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Rust Self-Play Engine                           │
│                                                                        │
│   [MCTS Actors 1..N] ──(EvalRequest)──► [Inference Dispatcher]         │
│           │                                     │                      │
│           │ (Buffered Steps)                    ▼                      │
│           ▼                              ort::Session (GPU)            │
│   [Trajectory Spooler]                          ▲                      │
│           │                                     │ Atomic Reload        │
│           ▼ (.bin files)                        │                      │
│   spool/traj_*.bin                      models/latest.onnx             │
└───────────┼─────────────────────────────────────▲──────────────────────┘
            │                                     │
            ▼                                     │ Atomic Export
┌─────────────────────────────────────────────────┴──────────────────────┐
│                        Python PyTorch Learner                          │
│                                                                        │
│   [Spool Reader (np.frombuffer)] ──► [Replay Buffer] ──► [PyTorch SGD] │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Invariants & Crate Boundaries

In strict compliance with [`AGENTS.md`](file:///home/pankaj/Projects/ai/AGENTS.md):
- `mcts-traits` **must never** depend on `ort`, `ndarray`, `notify`, `flume`, or `mcts-engine`.
- `mcts-engine` **must never** link against `ort` or neural network runtime libraries.
- All ONNX runtime dependencies, batch queuing, and file streaming belong exclusively in the new **`mcts-onnx`** crate.
- Game environment crates (`connect4`, `hex`, etc.) remain zero-dependency relative to `mcts-onnx`, implementing the generic `TensorRepresentable` trait from `mcts-traits`.

---

## 3. Detailed Component Architecture

### 3.1 `mcts-traits`: Tensor Representation Trait

Define `TensorRepresentable` in `mcts-traits/src/model.rs`:

```rust
/// Trait for environment states that can be encoded into planar float tensors (NCHW).
pub trait TensorRepresentable {
    /// Number of feature channels (e.g. 2 for Connect 4: own pieces, opponent pieces).
    const CHANNELS: usize;
    /// Spatial height dimension.
    const HEIGHT: usize;
    /// Spatial width dimension.
    const WIDTH: usize;

    /// Flattens the state into a pre-allocated float buffer of size `CHANNELS * HEIGHT * WIDTH`.
    ///
    /// Must write strictly to `out` without allocating heap memory.
    fn encode_tensor(&self, out: &mut [f32]);
}
```

### 3.2 `mcts-onnx`: Inference Dispatcher & Dynamic Micro-Batcher

Located in `mcts-onnx/src/inference.rs`:
- Receives `EvalRequest` items from search actor threads over a lockless `flume` channel.
- Accumulates items dynamically until reaching `max_batch_size` (e.g. 64 or 128) OR `max_latency` timeout (e.g. 2.0 ms).
- Packs all requests contiguously into an `ndarray::Array4<f32>` tensor.
- Runs `ort::Session::run()` once, slicing policy and value output tensors along `Axis(0)`.
- Dispatches results back to workers via one-shot channels (`tokio::sync::oneshot` or pre-allocated rendezvous slots).

### 3.3 `mcts-onnx`: Atomic Model Hot-Swapper

Located in `mcts-onnx/src/watcher.rs`:
- Uses the `notify` crate to monitor `models/latest.onnx`.
- When an atomic directory rename occurs (`Create` / `Modify` event), compiles the new `ort::Session` on a background thread.
- Performs an $O(1)$ atomic pointer swap inside the inference dispatcher between micro-batches without stalling MCTS workers.

### 3.4 `mcts-onnx`: Binary Trajectory Spooler Wire Protocol

Located in `mcts-onnx/src/spool.rs`:
- Fixed 64-byte header followed by contiguous, fixed-stride step records:

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
│                    CONTIGUOUS STEP RECORDS                   │
├──────────────────────────────────────────────────────────────┤
│ Step 0:                                                      │
│   observation:      [f32; C * H * W]                         │
│   action_mask:      [u8; (action_dim + 7) / 8]               │
│   policy_target:    [f32; action_dim]   (visit distribution) │
│   value_target:     [f32; num_players]  (game return vector) │
│   action_taken:     u32                                      │
│   reward_target:    [f32; num_players]  (intermediate return)│
├──────────────────────────────────────────────────────────────┤
│ Step 1: ...                                                  │
└──────────────────────────────────────────────────────────────┘
```

- Each worker buffers up to `CHUNK_SIZE` steps (e.g. 1,000 steps $\approx 1.5\text{–}3\,\text{MB}$).
- Writes to `spool/traj_<worker>_<ts>.bin.tmp`, flushes, and renames atomically to `.bin`.

### 3.5 Python Learner (`python/learner/`)

- `spool_reader.py`: Ingests chunk files via `np.frombuffer` with zero intermediate copies, extracting observation, policy, and value arrays.
- `model.py`: AlphaZero PyTorch ConvNet with residual blocks, policy head (softmax logits), and value head (tanh vector).
- `trainer.py`: Main training loop sampling mini-batches ($B = 256$), computing AlphaZero loss:
  $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( \|\mathbf{v}_\theta(s_i) - \mathbf{z}_i\|_2^2 - \boldsymbol{\pi}_i^\top \log \mathbf{p}_\theta(s_i) \right) + c \|\theta\|_2^2$$
  Periodically writes weights to `models/latest.onnx.tmp` and executes atomic swap `os.replace`.

---

## 4. Comprehensive Unit & Integration Test Plan

### 4.1 `mcts-traits` Tests
- `test_tensor_representable_contract`: Verify `encode_tensor` correctly populates exact buffer slices matching `CHANNELS * HEIGHT * WIDTH`.

### 4.2 `mcts-onnx` Wire Format & Spooling Tests
- `test_spooler_binary_roundtrip`:
  - Serialize 100 synthetic trajectory steps in Rust.
  - Read back using Rust binary parser, verifying bit-for-bit equality across header and step fields.
- `test_spooler_atomic_rename`:
  - Assert that `.bin.tmp` files are never visible to readers until flush and rename are complete.
- `test_spooler_fixed_stride_indexing`:
  - Verify that arbitrary step $k$ can be directly sliced at offset $64 + k \cdot \text{RecordStride}$.

### 4.3 `mcts-onnx` Inference Dispatcher Tests
- `test_dispatcher_single_request`: Send 1 request, verify correct shape and response reception.
- `test_dispatcher_micro_batch_accumulation`: Send 16 concurrent requests, assert that `session.run()` is called with batch dimension 16.
- `test_dispatcher_timeout_flush`: Send 1 request with a large batch limit; assert the request completes within `max_latency` timeout without waiting for more items.
- `test_model_watcher_hot_swap`:
  - Initialize session with dummy ONNX model V1.
  - Atomically replace file with dummy ONNX model V2.
  - Verify dispatcher transitions to V2 without panic or dropped queries.

### 4.4 Python Ingestion & Loss Tests (`python/learner/tests/`)
- `test_python_spool_reader`:
  - Read binary chunk generated by Rust spooler using `spool_reader.read_trajectory_chunk`.
  - Assert array dimensions, data types (`float32`), and values match original Rust records.
- `test_alphazero_loss_gradient_flow`:
  - Compute AlphaZero policy and value loss on dummy tensors, assert backpropagation produces non-zero gradients across all network parameters.
- `test_atomic_onnx_export`:
  - Export PyTorch model to ONNX, verify valid dynamic batch axes `(batch_size, ...)`.

### 4.5 End-to-End AlphaZero Integration Test (Connect 4)
- Run self-play generator `connect4-selfplay` for 200 games.
- Train Python learner on generated trajectory chunks for 500 gradient steps.
- Export `latest.onnx`.
- Benchmark trained model against uniform random baseline over 50 games in `connect4-tournament`; verify win rate $\ge 80\%$.

---

## 5. Implementation Roadmap & Checklist

- [ ] **Phase 1: `mcts-traits` Tensor Abstractions**
  - [ ] Add `TensorRepresentable` trait to `mcts-traits/src/model.rs`.
  - [ ] Implement `TensorRepresentable` for `Connect4State<R, C>`.
  - [ ] Implement `TensorRepresentable` for `HexState<N>`.

- [ ] **Phase 2: `mcts-onnx` Crate & Binary Spooler**
  - [ ] Initialize `mcts-onnx` crate with `ort`, `flume`, `ndarray`, `notify`.
  - [ ] Implement `TrajectorySpooler` with 64-byte header and fixed-stride binary writer.
  - [ ] Add unit tests verifying serialization and roundtrip integrity.

- [ ] **Phase 3: Dynamic Micro-Batcher & Hot-Swap Watcher**
  - [ ] Implement `InferenceDispatcher` and `OnnxModelClient`.
  - [ ] Implement `start_model_watcher` using `notify`.
  - [ ] Add concurrent unit tests with synthetic ONNX test models.

- [ ] **Phase 4: Python Learner Service**
  - [ ] Implement `python/learner/spool_reader.py` with zero-copy numpy slicing.
  - [ ] Implement `python/learner/model.py` (AlphaZero dual-headed ConvNet).
  - [ ] Implement `python/learner/trainer.py` with replay buffer and atomic ONNX export.

- [ ] **Phase 5: Self-Play CLI & End-to-End Verification**
  - [ ] Create `connect4-selfplay` binary in `environments/connect4/src/bin/selfplay.rs`.
  - [ ] Execute self-play $\to$ training $\to$ hot-swap $\to$ evaluation pipeline.
  - [ ] Validate throughput: $>5,000$ leaf evals/sec on CPU / $>50,000$ on GPU.
  - [ ] Verify `cargo check --workspace`, `cargo test --workspace`, and `cargo clippy` pass with 0 warnings.

---

## 6. Acceptance Criteria

1. **Decoupled Invariants**: `mcts-traits` and `mcts-engine` remain completely free of `ort` or neural network framework dependencies.
2. **Zero Inode Churn**: Binary chunks are flushed in fixed 1,000-step batches with atomic rename, maintaining filesystem hygiene.
3. **Zero Search Stall during Weights Reload**: Background compilation and atomic pointer replacement swap ONNX sessions in $<150\,\text{ms}$ with zero dropped search queries.
4. **Clean Verification**: All unit tests pass across Rust and Python test runners with zero compiler warnings or clippy lints.
