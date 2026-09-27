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
