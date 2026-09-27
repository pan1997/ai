"""Tabular AlphaZero model for exact verification testing on discrete MDPs."""

import os
import struct
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn

TABULAR_MAGIC = b"TABL"


class TabularAlphaZero(nn.Module):
    """Tabular AlphaZero parameter lookup table for discrete state MDPs."""

    def __init__(self, num_states: int, num_actions: int, num_players: int = 1):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_players = num_players

        # Trainable parameters initialized to zero
        self.policy_table = nn.Parameter(torch.zeros(num_states, num_actions))
        self.value_table = nn.Parameter(torch.zeros(num_states, num_players))

    def forward(self, state_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Looks up policy logits and value estimates for a batch of state indices.

        Args:
            state_ids: 1D LongTensor of shape `(batch_size,)`.

        Returns:
            policy_logits: `(batch_size, num_actions)`
            values: `(batch_size, num_players)`
        """
        logits = self.policy_table[state_ids]
        values = self.value_table[state_ids]
        return logits, values

    def save_binary(self, save_path: Path | str) -> None:
        """Saves tabular weights in the portable binary format (`TABL`) with atomic rename."""
        path = Path(save_path)
        tmp_path = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as f:
            f.write(TABULAR_MAGIC)
            f.write(struct.pack("<IIII", 1, self.num_states, self.num_actions, self.num_players))

            policy_data = self.policy_table.detach().cpu().numpy()
            value_data = self.value_table.detach().cpu().numpy()

            for s in range(self.num_states):
                f.write(struct.pack("<I", s))
                f.write(policy_data[s].astype("<f4").tobytes())
                f.write(value_data[s].astype("<f4").tobytes())

            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    @classmethod
    def load_binary(cls, path: Path | str) -> "TabularAlphaZero":
        """Loads tabular weights from a binary file."""
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != TABULAR_MAGIC:
                raise ValueError(f"Invalid magic: {magic}")
            version, num_states, num_actions, num_players = struct.unpack("<IIII", f.read(16))
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")

            model = cls(num_states, num_actions, num_players)
            policy_buf = torch.zeros(num_states, num_actions)
            value_buf = torch.zeros(num_states, num_players)

            for _ in range(num_states):
                (state,) = struct.unpack("<I", f.read(4))
                p_bytes = f.read(4 * num_actions)
                v_bytes = f.read(4 * num_players)
                policy_buf[state] = torch.tensor(struct.unpack(f"<{num_actions}f", p_bytes))
                value_buf[state] = torch.tensor(struct.unpack(f"<{num_players}f", v_bytes))

            model.policy_table.data.copy_(policy_buf)
            model.value_table.data.copy_(value_buf)
            return model
