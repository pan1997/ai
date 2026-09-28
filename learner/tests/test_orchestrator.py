import tempfile
import time
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

    # Also test Connect 4 display format
    c4_output = """
==========================================================================================
                                    FINAL STANDINGS                                       
==========================================================================================
Pos  Agent                           Games   Wins   Loss   Draw    Win %     Red W%     Yel W%
--------------------------------------------------------------------------------------------
1    AlphaZero(50)                      20     18      1      1    90.0%      90.0%      90.0%
2    Tactical                           20      1     18      1     5.0%       5.0%       5.0%
==========================================================================================
"""
    c4_parsed = evaluator._parse_standings(c4_output, "alphazero")
    assert c4_parsed is not None
    assert c4_parsed["games"] == 20
    assert c4_parsed["wins"] == 18
    assert c4_parsed["loss"] == 1
    assert c4_parsed["draw"] == 1
    assert c4_parsed["win_rate"] == 0.9


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


def test_updates_per_transition_orchestration():
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        spool_dir = root_dir / "spool"
        spool_dir.mkdir(parents=True)
        model_path = root_dir / "latest.onnx"
        ckpt_path = root_dir / "latest.pt"
        db_path = root_dir / "mlflow.db"

        import struct
        from learner.spool_reader import HEADER_FORMAT, MAGIC
        chunk_file = spool_dir / "traj_init.bin"
        num_steps = 30
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

        # updates_per_transition = 0.5 -> 30 steps * 0.5 = 15 gradient steps!
        orchestrator = ExperimentOrchestrator(
            spool_dir=str(spool_dir),
            model_path=str(model_path),
            checkpoint_path=str(ckpt_path),
            tracking_uri=f"sqlite:///{db_path}",
            experiment_name="test_updates_per_transition",
            run_name="test_rate_run",
            min_new_steps=10,
            batch_size=8,
            updates_per_transition=0.5,
            poll_interval=0.1,
            eval_interval=0,
            enable_selfplay=False,
        )

        orchestrator.run(max_iterations=1)

        assert orchestrator.iteration == 1
        assert orchestrator.global_step == 15  # Exactly 30 * 0.5 = 15 updates!


def test_selfplay_supervisor_multiple_workers():
    import sys
    from learner.orchestrator import SelfPlaySupervisor

    supervisor = SelfPlaySupervisor(
        binary_path=sys.executable,
        spool_dir="/tmp",
        model_path="/tmp/model",
        num_workers=3,
        games_per_batch=1,
        sims=1,
    )
    supervisor.start()
    assert len(supervisor.worker_threads) == 3
    supervisor.stop()
    assert len(supervisor._processes) == 0


def test_selfplay_supervisor_backpressure():
    import sys
    from learner.orchestrator import SelfPlaySupervisor

    with tempfile.TemporaryDirectory() as tmpdir:
        spool_dir = Path(tmpdir)
        (spool_dir / "traj_001.bin").write_bytes(b"dummy")
        (spool_dir / "traj_002.bin").write_bytes(b"dummy")

        supervisor = SelfPlaySupervisor(
            binary_path=sys.executable,
            spool_dir=str(spool_dir),
            model_path="/tmp/model",
            num_workers=1,
            games_per_batch=1,
            sims=1,
            max_spool_chunks=2,
        )
        supervisor.start()
        # Give worker thread a moment to run its loop
        time.sleep(0.2)
        # Should be paused by backpressure because pending_chunks (2) >= max_spool_chunks (2)
        assert len(supervisor._processes) == 0

        # Unlink one file to relieve backpressure
        (spool_dir / "traj_001.bin").unlink()
        time.sleep(0.3)
        # Once relieved, worker should attempt to spawn
        # (It will fail quickly since sys.executable isn't a valid selfplay binary, but the process was created/attempted)
        supervisor.stop()



def test_game_config_registry():
    from learner.game_config import get_game_config, GAME_REGISTRY

    ttt = get_game_config("tictactoe")
    assert ttt.name == "tictactoe"
    assert ttt.height == 3 and ttt.width == 3 and ttt.action_dim == 9

    c4 = get_game_config("connect4")
    assert c4.name == "connect4"
    assert c4.height == 6 and c4.width == 7 and c4.action_dim == 7

    with pytest.raises(ValueError, match="Unknown game"):
        get_game_config("nonexistent_game")


def test_orchestrator_connect4_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        orchestrator = ExperimentOrchestrator(
            game="connect4",
            spool_dir=str(root_dir / "spool"),
            model_path=str(root_dir / "latest.onnx"),
            checkpoint_path=str(root_dir / "latest.pt"),
            tracking_uri=f"sqlite:///{root_dir / 'mlflow.db'}",
            enable_selfplay=False,
        )
        assert orchestrator.game == "connect4"
        assert orchestrator.height == 6
        assert orchestrator.width == 7
        assert orchestrator.action_dim == 7
        assert orchestrator.model.policy_fcs[0].out_features == 7


def test_ingest_corrupted_or_empty_spool_chunk():
    from learner.trainer import ingest_spool_chunks

    with tempfile.TemporaryDirectory() as tmpdir:
        spool_dir = Path(tmpdir)
        buffer = ReplayBuffer(max_capacity=50)

        # 0-byte file
        (spool_dir / "traj_empty.bin").write_bytes(b"")
        # Incomplete header (< 64 bytes)
        (spool_dir / "traj_partial.bin").write_bytes(b"MCTS" + b"\x00" * 10)

        # Ingestion should ignore/skip corrupt files without raising uncaught exceptions
        ingested = ingest_spool_chunks(spool_dir, buffer)
        assert ingested == 0
        assert len(buffer) == 0


