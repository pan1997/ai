import struct
import tempfile
from pathlib import Path
import numpy as np

from learner.spool_reader import HEADER_FORMAT, MAGIC
from learner.trainer import ReplayBuffer, ingest_spool_chunks


def test_replay_buffer_capacity_and_sampling():
    buffer = ReplayBuffer(max_capacity=10)

    dt = np.dtype([
        ("obs", np.float32, (4,)),
        ("mask", np.uint8, (4,)),
        ("policy", np.float32, (2,)),
        ("value", np.float32, (1,)),
        ("action", np.uint32),
        ("reward", np.float32, (1,)),
    ])

    batch1 = np.zeros(6, dtype=dt)
    for i in range(6):
        batch1[i]["action"] = i

    buffer.extend(batch1)
    assert len(buffer) == 6

    batch2 = np.zeros(6, dtype=dt)
    for i in range(6):
        batch2[i]["action"] = 10 + i

    buffer.extend(batch2)
    assert len(buffer) == 10  # Capped at 10

    # Actions 0 and 1 should have been evicted; remaining should be 2..5 and 10..15
    actions = [rec["action"] for rec in buffer.buffer]
    assert actions == [2, 3, 4, 5, 10, 11, 12, 13, 14, 15]

    sample = buffer.sample_batch(4)
    assert len(sample) == 4


def test_ingest_spool_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        spool_dir = Path(tmpdir)
        chunk_file = spool_dir / "traj_001.bin"

        num_steps = 3
        channels = 1
        height = 2
        width = 2
        action_dim = 2
        num_players = 1

        obs_len = 4
        mask_len = 4
        policy_len = 2
        val_len = 1
        stride_bytes = 4 * 4 + 4 + 2 * 4 + 1 * 4 + 4 + 1 * 4

        header_bytes = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            1,
            0,
            num_steps,
            0,
            channels,
            height,
            width,
            action_dim,
            num_players,
            stride_bytes,
            b"\x00" * 16,
        )

        dt = np.dtype([
            ("obs", np.float32, (obs_len,)),
            ("mask", np.uint8, (mask_len,)),
            ("policy", np.float32, (policy_len,)),
            ("value", np.float32, (val_len,)),
            ("action", np.uint32),
            ("reward", np.float32, (val_len,)),
        ])

        records = np.zeros(num_steps, dtype=dt)
        with open(chunk_file, "wb") as f:
            f.write(header_bytes)
            f.write(records.tobytes())

        buffer = ReplayBuffer(max_capacity=100)
        ingested = ingest_spool_chunks(spool_dir, buffer)

        assert ingested == 3
        assert len(buffer) == 3
        # Chunk file must be unlinked/deleted after ingestion
        assert not chunk_file.exists()


def test_ingest_spool_chunks_max_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        spool_dir = Path(tmpdir)
        num_steps = 3
        channels, height, width, action_dim, num_players = 1, 2, 2, 2, 1
        obs_len, mask_len, policy_len, val_len = 4, 4, 2, 1
        stride_bytes = 4 * 4 + 4 + 2 * 4 + 1 * 4 + 4 + 1 * 4

        header_bytes = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            1,
            0,
            num_steps,
            0,
            channels,
            height,
            width,
            action_dim,
            num_players,
            stride_bytes,
            b"\x00" * 16,
        )
        dt = np.dtype([
            ("obs", np.float32, (obs_len,)),
            ("mask", np.uint8, (mask_len,)),
            ("policy", np.float32, (policy_len,)),
            ("value", np.float32, (val_len,)),
            ("action", np.uint32),
            ("reward", np.float32, (val_len,)),
        ])
        records = np.zeros(num_steps, dtype=dt)

        file1 = spool_dir / "traj_001.bin"
        file2 = spool_dir / "traj_002.bin"
        with open(file1, "wb") as f:
            f.write(header_bytes)
            f.write(records.tobytes())
        with open(file2, "wb") as f:
            f.write(header_bytes)
            f.write(records.tobytes())

        buffer = ReplayBuffer(max_capacity=100)
        # Ingest with max_chunks=1
        ingested = ingest_spool_chunks(spool_dir, buffer, max_chunks=1)
        assert ingested == 3
        assert len(buffer) == 3
        # file1 unlinked, but file2 remains
        assert not file1.exists()
        assert file2.exists()

        # Next ingestion ingests file2
        ingested2 = ingest_spool_chunks(spool_dir, buffer, max_chunks=1)
        assert ingested2 == 3
        assert len(buffer) == 6
        assert not file2.exists()

