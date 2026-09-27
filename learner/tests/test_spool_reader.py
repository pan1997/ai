import struct
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from learner.spool_reader import HEADER_FORMAT, HEADER_SIZE, MAGIC, read_trajectory_chunk


def test_read_trajectory_chunk_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir) / "traj_00001.bin"

        num_steps = 5
        channels = 2
        height = 3
        width = 3
        action_dim = 9
        num_players = 2

        obs_len = channels * height * width
        mask_len = ((action_dim + 31) // 32) * 4
        policy_len = action_dim
        val_len = num_players

        stride_bytes = obs_len * 4 + mask_len + policy_len * 4 + val_len * 4 + 4 + val_len * 4

        header_bytes = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            1,  # version
            0,  # game_id
            num_steps,
            0,  # obs_dtype
            channels,
            height,
            width,
            action_dim,
            num_players,
            stride_bytes,
            b"\x00" * 16,  # reserved (16 bytes)
        )
        assert len(header_bytes) == HEADER_SIZE

        dt = np.dtype([
            ("obs", np.float32, (obs_len,)),
            ("mask", np.uint8, (mask_len,)),
            ("policy", np.float32, (policy_len,)),
            ("value", np.float32, (val_len,)),
            ("action", np.uint32),
            ("reward", np.float32, (val_len,)),
        ])

        records = np.zeros(num_steps, dtype=dt)
        for i in range(num_steps):
            records[i]["obs"] = np.full(obs_len, float(i), dtype=np.float32)
            records[i]["policy"] = np.full(policy_len, 1.0 / policy_len, dtype=np.float32)
            records[i]["value"] = np.array([1.0, -1.0], dtype=np.float32)
            records[i]["action"] = i
            records[i]["reward"] = np.array([0.0, 0.0], dtype=np.float32)

        with open(chunk_path, "wb") as f:
            f.write(header_bytes)
            f.write(records.tobytes())

        # Read using spool reader
        header, loaded_records = read_trajectory_chunk(chunk_path)

        assert header["num_steps"] == num_steps
        assert header["channels"] == channels
        assert header["action_dim"] == action_dim
        assert len(loaded_records) == num_steps

        # Verify zero-copy Torch compatibility due to 4-byte natural alignment
        t_obs = torch.from_numpy(loaded_records["obs"])
        t_policy = torch.from_numpy(loaded_records["policy"])
        t_value = torch.from_numpy(loaded_records["value"])

        assert t_obs.shape == (num_steps, obs_len)
        assert t_policy.shape == (num_steps, policy_len)
        assert t_value.shape == (num_steps, num_players)
        assert float(t_obs[2, 0]) == 2.0


def test_invalid_magic_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir) / "invalid.bin"
        with open(chunk_path, "wb") as f:
            f.write(b"BAD!" + b"\x00" * 60)

        with pytest.raises(ValueError, match="Invalid magic"):
            read_trajectory_chunk(chunk_path)
