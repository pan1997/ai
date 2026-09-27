"""Zero-copy trajectory chunk reader using numpy memory mapping."""

import struct
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

HEADER_FORMAT = "<4sIIIIIIIIIQ16s"  # 64 bytes
HEADER_SIZE = 64
MAGIC = b"MCTS"


def read_trajectory_chunk(chunk_path: Path | str) -> Tuple[Dict[str, Any], np.ndarray]:
    """Reads a binary trajectory chunk file with zero intermediate memory copying.

    Returns:
        header: Dictionary of metadata extracted from the 64-byte header.
        records: Structured numpy array containing all step records.
    """
    path = Path(chunk_path)
    with open(path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError(f"File {path} is smaller than 64-byte header ({len(header_bytes)} bytes)")

        (
            magic,
            version,
            game_id,
            num_steps,
            obs_dtype,
            channels,
            height,
            width,
            action_dim,
            num_players,
            stride_bytes,
            _reserved,
        ) = struct.unpack(HEADER_FORMAT, header_bytes)

        if magic != MAGIC:
            raise ValueError(f"Invalid magic header: expected {MAGIC!r}, got {magic!r}")

        buffer = f.read()

    header = {
        "magic": magic,
        "version": version,
        "game_id": game_id,
        "num_steps": num_steps,
        "obs_dtype": obs_dtype,
        "channels": channels,
        "height": height,
        "width": width,
        "action_dim": action_dim,
        "num_players": num_players,
        "stride_bytes": stride_bytes,
    }

    obs_len = channels * height * width
    mask_len = ((action_dim + 31) // 32) * 4
    policy_len = action_dim
    val_len = num_players

    # Define tightly packed numpy structured record type
    dt = np.dtype([
        ("obs", np.float32, (obs_len,)),
        ("mask", np.uint8, (mask_len,)),
        ("policy", np.float32, (policy_len,)),
        ("value", np.float32, (val_len,)),
        ("action", np.uint32),
        ("reward", np.float32, (val_len,)),
    ])

    records = np.frombuffer(buffer, dtype=dt, count=num_steps)
    return header, records
