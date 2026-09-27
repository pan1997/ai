import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from learner.models.convnet import AlphaZeroConvNet
from learner.orchestrator import ExperimentOrchestrator, TournamentEvaluator
from learner.trainer import AlphaZeroTrainer, ReplayBuffer


def test_alphazero_trainer_lifecycle_and_checkpoints():
    in_channels = 3
    height = 3
    width = 3
    action_dim = 9
    num_players = 2

    model = AlphaZeroConvNet(in_channels, height, width, action_dim, num_players, hidden_channels=16, num_res_blocks=1)
    buffer = ReplayBuffer(max_capacity=50)

    # Populate dummy replay buffer
    dt = np.dtype([
        ("obs", np.float32, (in_channels * height * width,)),
        ("mask", np.uint8, (4,)),
        ("policy", np.float32, (action_dim,)),
        ("value", np.float32, (num_players,)),
        ("action", np.uint32),
        ("reward", np.float32, (num_players,)),
    ])
    records = np.zeros(20, dtype=dt)
    for i in range(20):
        records[i]["policy"] = np.ones(action_dim) / action_dim
        records[i]["value"] = np.array([1.0, -1.0])
    buffer.extend(records)

    trainer = AlphaZeroTrainer(
        model=model,
        replay_buffer=buffer,
        learning_rate=0.005,
        in_channels=in_channels,
        height=height,
        width=width,
    )

    tot, p, v, steps = trainer.train_iteration(epochs=2, batch_size=8)
    assert steps > 0
    assert tot > 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "latest.pt"
        onnx_path = Path(tmpdir) / "latest.onnx"
        meta = {"iteration": 5, "global_step": 100, "mlflow_run_id": "test_123"}

        trainer.save_checkpoint(ckpt_path, onnx_path, metadata=meta)
        assert ckpt_path.exists()
        assert onnx_path.exists()
        assert ckpt_path.with_suffix(".meta.json").exists()

        # Create a new trainer and load checkpoint
        new_model = AlphaZeroConvNet(in_channels, height, width, action_dim, num_players, hidden_channels=16, num_res_blocks=1)
        new_trainer = AlphaZeroTrainer(new_model, ReplayBuffer(10), in_channels=in_channels, height=height, width=width)
        loaded_meta = new_trainer.load_checkpoint(ckpt_path)

        assert loaded_meta["iteration"] == 5
        assert loaded_meta["global_step"] == 100
        assert loaded_meta["mlflow_run_id"] == "test_123"


def test_tournament_evaluator_standings_parsing():
    evaluator = TournamentEvaluator(binary_path="/nonexistent")
    sample_output = """
Tournament Completed in 3.16s!

==========================================================================================
                                    FINAL STANDINGS                                       
==========================================================================================
Pos  Agent                           Games   Wins   Loss   Draw    Win %       X W%       O W%
--------------------------------------------------------------------------------------------
1    alphazero:./models/latest.onnx:50     10      0      0     10     0.0%       0.0%       0.0%
2    tactical                           10      0      0     10     0.0%       0.0%       0.0%
==========================================================================================
"""
    parsed = evaluator._parse_standings(sample_output, "alphazero")
    assert parsed is not None
    assert parsed["games"] == 10
    assert parsed["wins"] == 0
    assert parsed["loss"] == 0
    assert parsed["draw"] == 10
    assert parsed["draw_rate"] == 1.0


def test_experiment_orchestrator_short_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        spool_dir = root_dir / "spool"
        spool_dir.mkdir(parents=True)
        model_path = root_dir / "latest.onnx"
        ckpt_path = root_dir / "latest.pt"
        db_path = root_dir / "mlflow.db"

        # Populate pre-ingested records
        import struct
        from learner.spool_reader import HEADER_FORMAT, MAGIC
        chunk_file = spool_dir / "traj_init.bin"
        num_steps = 15
        channels = 3
        height = 3
        width = 3
        action_dim = 9
        num_players = 2

        obs_len = channels * height * width
        mask_len = 4
        stride_bytes = obs_len * 4 + mask_len + action_dim * 4 + num_players * 4 + 4 + num_players * 4

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
            ("policy", np.float32, (action_dim,)),
            ("value", np.float32, (num_players,)),
            ("action", np.uint32),
            ("reward", np.float32, (num_players,)),
        ])
        records = np.zeros(num_steps, dtype=dt)
        for i in range(num_steps):
            records[i]["policy"] = np.ones(action_dim) / action_dim
            records[i]["value"] = np.array([1.0, -1.0])
        with open(chunk_file, "wb") as f:
            f.write(header_bytes)
            f.write(records.tobytes())

        orchestrator = ExperimentOrchestrator(
            spool_dir=str(spool_dir),
            model_path=str(model_path),
            checkpoint_path=str(ckpt_path),
            tracking_uri=f"sqlite:///{db_path}",
            experiment_name="test_orchestrator",
            run_name="test_run",
            min_new_steps=5,
            epochs_per_iter=2,
            batch_size=8,
            poll_interval=0.1,
            eval_interval=0,
            enable_selfplay=False,
        )

        # Run for 1 iteration
        orchestrator.run(max_iterations=1)

        assert orchestrator.iteration == 1
        assert model_path.exists()
        assert ckpt_path.exists()
