"""AlphaZero loss functions, replay buffer, and autonomous trainer service."""

import os
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learner.models.convnet import export_alphazero_onnx
from learner.spool_reader import read_trajectory_chunk


def compute_alphazero_loss(
    pred_policy_logits: torch.Tensor,
    pred_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the dual-headed AlphaZero multi-task loss.

    Formula:
        L_policy = - sum(pi * log_softmax(p)) / B
        L_value  = mean(||v - z||^2)
        L_total  = L_policy + L_value

    Returns:
        total_loss, policy_loss, value_loss
    """
    # Policy cross-entropy loss over action probabilities
    log_probs = F.log_softmax(pred_policy_logits, dim=-1)
    policy_loss = -torch.sum(target_policy * log_probs, dim=-1).mean()

    # Value mean squared error
    value_loss = F.mse_loss(pred_value, target_value)

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss


class ReplayBuffer:
    """In-memory experience replay buffer with FIFO eviction."""

    def __init__(self, max_capacity: int = 100_000):
        self.max_capacity = max_capacity
        self.buffer: List[np.ndarray] = []

    def extend(self, records: np.ndarray) -> None:
        """Appends a structured array of records, evicting oldest if capacity is exceeded."""
        self.buffer.extend(records)
        if len(self.buffer) > self.max_capacity:
            excess = len(self.buffer) - self.max_capacity
            self.buffer = self.buffer[excess:]

    def sample_batch(self, batch_size: int) -> List[np.ndarray]:
        """Samples a uniform random mini-batch without replacement."""
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


def ingest_spool_chunks(spool_dir: Path | str, replay_buffer: ReplayBuffer) -> int:
    """Scans `spool_dir` for completed `.bin` chunks, ingests them into the buffer, and unlinks them.

    Returns:
        Total number of steps ingested in this sweep.
    """
    path = Path(spool_dir)
    if not path.exists():
        return 0

    total_ingested = 0
    for chunk_file in sorted(path.glob("traj_*.bin")):
        try:
            _header, records = read_trajectory_chunk(chunk_file)
            replay_buffer.extend(records)
            total_ingested += len(records)
            chunk_file.unlink()
        except (IOError, ValueError, PermissionError) as e:
            # Chunk may be partially written or locked; retry next sweep
            continue

    return total_ingested


class AlphaZeroTrainer:
    """Stateful trainer managing neural network weights, optimizer, and training updates."""

    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-4,
        in_channels: int = 3,
        height: int = 3,
        width: int = 3,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.replay_buffer = replay_buffer
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def ingest(self, spool_dir: Path | str) -> int:
        """Ingests completed .bin trajectory chunks from spool_dir into the replay buffer."""
        return ingest_spool_chunks(spool_dir, self.replay_buffer)

    def train_epoch(self, batch_size: int) -> Tuple[float, float, float, int]:
        """Runs 1 training epoch over the replay buffer.

        Returns:
            (avg_total_loss, avg_policy_loss, avg_value_loss, num_gradient_steps)
        """
        if len(self.replay_buffer) < 10:
            return 0.0, 0.0, 0.0, 0

        self.model.train()
        actual_batch_size = min(batch_size, len(self.replay_buffer))
        steps = max(1, len(self.replay_buffer) // actual_batch_size)

        losses = []
        for _ in range(steps):
            batch = self.replay_buffer.sample_batch(actual_batch_size)
            obs_batch = np.stack([r["obs"] for r in batch]).reshape(
                -1, self.in_channels, self.height, self.width
            )
            policy_batch = np.stack([r["policy"] for r in batch])
            value_batch = np.stack([r["value"] for r in batch])

            x = torch.from_numpy(obs_batch).float().to(self.device)
            target_p = torch.from_numpy(policy_batch).float().to(self.device)
            target_v = torch.from_numpy(value_batch).float().to(self.device)

            self.optimizer.zero_grad()
            p_logits, v_pred = self.model(x)
            total_loss, p_loss, v_loss = compute_alphazero_loss(
                p_logits, v_pred, target_p, target_v
            )
            total_loss.backward()
            self.optimizer.step()

            losses.append((total_loss.item(), p_loss.item(), v_loss.item()))

        avg_tot = float(np.mean([l[0] for l in losses]))
        avg_p = float(np.mean([l[1] for l in losses]))
        avg_v = float(np.mean([l[2] for l in losses]))
        return avg_tot, avg_p, avg_v, len(losses)

    def train_steps(
        self, num_steps: int, batch_size: int
    ) -> Tuple[float, float, float, int]:
        """Runs an explicit number of mini-batch gradient optimization steps over the replay buffer.

        Returns:
            (avg_total_loss, avg_policy_loss, avg_value_loss, steps_executed)
        """
        if len(self.replay_buffer) < 10 or num_steps <= 0:
            return 0.0, 0.0, 0.0, 0

        self.model.train()
        actual_batch_size = min(batch_size, len(self.replay_buffer))

        losses = []
        for _ in range(num_steps):
            batch = self.replay_buffer.sample_batch(actual_batch_size)
            obs_batch = np.stack([r["obs"] for r in batch]).reshape(
                -1, self.in_channels, self.height, self.width
            )
            policy_batch = np.stack([r["policy"] for r in batch])
            value_batch = np.stack([r["value"] for r in batch])

            x = torch.from_numpy(obs_batch).float().to(self.device)
            target_p = torch.from_numpy(policy_batch).float().to(self.device)
            target_v = torch.from_numpy(value_batch).float().to(self.device)

            self.optimizer.zero_grad()
            p_logits, v_pred = self.model(x)
            total_loss, p_loss, v_loss = compute_alphazero_loss(
                p_logits, v_pred, target_p, target_v
            )
            total_loss.backward()
            self.optimizer.step()

            losses.append((total_loss.item(), p_loss.item(), v_loss.item()))

        avg_tot = float(np.mean([l[0] for l in losses]))
        avg_p = float(np.mean([l[1] for l in losses]))
        avg_v = float(np.mean([l[2] for l in losses]))
        return avg_tot, avg_p, avg_v, len(losses)

    def train_iteration(
        self, epochs: int, batch_size: int
    ) -> Tuple[float, float, float, int]:
        """Runs multiple training epochs over the replay buffer.

        Returns:
            (avg_total_loss, avg_policy_loss, avg_value_loss, total_gradient_steps)
        """
        all_tot, all_p, all_v = [], [], []
        total_steps = 0
        for _ in range(epochs):
            tot, p, v, steps = self.train_epoch(batch_size)
            if steps > 0:
                all_tot.append(tot)
                all_p.append(p)
                all_v.append(v)
                total_steps += steps

        if not all_tot:
            return 0.0, 0.0, 0.0, 0

        return (
            float(np.mean(all_tot)),
            float(np.mean(all_p)),
            float(np.mean(all_v)),
            total_steps,
        )

    def save_checkpoint(
        self,
        checkpoint_path: Path | str,
        onnx_out_path: Path | str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Saves PyTorch weights, optimizer state, and optional ONNX model export."""
        import json

        ckpt_file = Path(checkpoint_path)
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        tmp_ckpt = ckpt_file.with_suffix(".tmp")
        torch.save(payload, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_file)

        meta_json_path = ckpt_file.with_suffix(".meta.json")
        if metadata:
            tmp_json = meta_json_path.with_suffix(".tmp")
            with open(tmp_json, "w") as f:
                json.dump(metadata, f, indent=2)
            os.replace(tmp_json, meta_json_path)

        if onnx_out_path:
            export_alphazero_onnx(
                self.model,
                onnx_out_path,
                (self.in_channels, self.height, self.width),
            )

    def load_checkpoint(self, checkpoint_path: Path | str) -> dict:
        """Loads weights and optimizer state from checkpoint.

        Supports both structured payload dicts and plain state_dict files.
        Returns the loaded metadata dictionary.
        """
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            return {}

        data = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        if isinstance(data, dict) and "model_state_dict" in data:
            self.model.load_state_dict(data["model_state_dict"])
            if "optimizer_state_dict" in data:
                try:
                    self.optimizer.load_state_dict(data["optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: could not restore optimizer state: {e}")
            return data.get("metadata", {})
        else:
            self.model.load_state_dict(data)
            return {}

