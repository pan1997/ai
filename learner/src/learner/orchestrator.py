"""Autonomous experiment orchestrator for live continuous AlphaZero learning with MLflow tracking.

Supervises background self-play workers, ingests spooled trajectories, executes in-process
neural network training iterations, hot-swaps ONNX models, and logs real-time metrics,
artifacts, and tournament evaluations to MLflow.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import torch

from learner.game_config import GameConfig, get_game_config
from learner.models.convnet import AlphaZeroConvNet
from learner.trainer import AlphaZeroTrainer, ReplayBuffer


class SelfPlaySupervisor:
    """Supervises continuous self-play background worker subprocesses."""

    def __init__(
        self,
        binary_path: str,
        spool_dir: str,
        model_path: str,
        num_workers: int = 1,
        games_per_batch: int = 50,
        sims: int = 40,
        c_puct: float = 1.414,
        parallel_games: int = 16,
        max_spool_chunks: int = 4,
    ):
        self.binary_path = binary_path
        self.spool_dir = spool_dir
        self.model_path = model_path
        self.num_workers = max(1, num_workers)
        self.games_per_batch = games_per_batch
        self.sims = sims
        self.c_puct = c_puct
        self.parallel_games = parallel_games
        self.max_spool_chunks = max(1, max_spool_chunks)
        self.stop_event = threading.Event()
        self.worker_threads: List[threading.Thread] = []
        self._processes: Dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the background self-play worker threads."""
        self.stop_event.clear()
        self.worker_threads.clear()
        for worker_id in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"SelfPlayWorker-{worker_id}",
                daemon=True,
            )
            t.start()
            self.worker_threads.append(t)
        print(
            f"[SelfPlaySupervisor] Started {self.num_workers} parallel self-play worker(s) using '{self.binary_path}'"
        )

    def _worker_loop(self, worker_id: int) -> None:
        cmd = [
            self.binary_path,
            "--spool-dir",
            self.spool_dir,
            "--model-path",
            self.model_path,
            "--games",
            str(self.games_per_batch),
            "--sims",
            str(self.sims),
            "--c-puct",
            str(self.c_puct),
            "--worker-id",
            str(worker_id),
            "--parallel-games",
            str(self.parallel_games),
        ]

        spool_path = Path(self.spool_dir)
        while not self.stop_event.is_set():
            # Check spool backpressure: pause worker if unconsumed chunks exceed limit
            if spool_path.exists():
                try:
                    pending_chunks = len(list(spool_path.glob("traj_*.bin")))
                    if pending_chunks >= self.max_spool_chunks:
                        time.sleep(0.1)
                        continue
                except OSError:
                    pass

            try:
                with self._lock:
                    if self.stop_event.is_set():
                        break
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    self._processes[worker_id] = proc

                stdout, stderr = proc.communicate()

                with self._lock:
                    self._processes.pop(worker_id, None)

                if proc.returncode != 0 and not self.stop_event.is_set():
                    print(
                        f"[SelfPlaySupervisor] Worker {worker_id} exited with code {proc.returncode}. Stderr: {stderr.strip()}"
                    )
                    time.sleep(2.0)
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[SelfPlaySupervisor] Error running worker {worker_id}: {e}")
                    time.sleep(2.0)

    def stop(self) -> None:
        """Stops all worker threads and terminates any running subprocesses cleanly."""
        print(f"[SelfPlaySupervisor] Stopping {len(self.worker_threads)} self-play worker(s)...")
        self.stop_event.set()
        with self._lock:
            for wid, proc in list(self._processes.items()):
                if proc.poll() is None:
                    proc.terminate()
            for wid, proc in list(self._processes.items()):
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            self._processes.clear()

        for t in self.worker_threads:
            t.join(timeout=5.0)
        print("[SelfPlaySupervisor] All self-play workers stopped.")



class TournamentEvaluator:
    """Runs periodic competitive evaluations against heuristic reference agents."""

    STANDINGS_REGEX = re.compile(
        r"^\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)%",
        re.MULTILINE,
    )

    def __init__(self, binary_path: str):
        self.binary_path = binary_path

    def evaluate(
        self,
        model_path: str,
        num_sims: int = 50,
        games: int = 20,
        baselines: Optional[list] = None,
    ) -> Dict[str, float]:
        """Runs tournament matches against specified baseline agents.

        Returns:
            Dictionary of metrics (e.g. draw rates, win rates).
        """
        metrics = {}
        if not Path(self.binary_path).exists():
            return metrics

        for baseline in (baselines or ["tactical", "random"]):
            cmd = [
                self.binary_path,
                "--p1",
                f"alphazero:{model_path}:{num_sims}",
                "--p2",
                baseline,
                "--games",
                str(games),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                stats = self._parse_standings(res.stdout, "alphazero")
                if stats:
                    metrics[f"eval/{baseline}_win_rate"] = stats["win_rate"]
                    metrics[f"eval/{baseline}_draw_rate"] = stats["draw_rate"]
                    metrics[f"eval/{baseline}_loss_rate"] = stats["loss_rate"]

        return metrics

    def _parse_standings(self, output: str, agent_prefix: str) -> Optional[Dict[str, float]]:
        for match in self.STANDINGS_REGEX.finditer(output):
            agent_name = match.group(2)
            if agent_prefix.lower() in agent_name.lower():
                games = int(match.group(3))
                wins = int(match.group(4))
                loss = int(match.group(5))
                draw = int(match.group(6))
                if games > 0:
                    return {
                        "games": games,
                        "wins": wins,
                        "loss": loss,
                        "draw": draw,
                        "win_rate": wins / games,
                        "draw_rate": draw / games,
                        "loss_rate": loss / games,
                    }
        return None


class ExperimentOrchestrator:
    """Manages the full lifecycle of continuous AlphaZero actor-learner pipelines with MLflow."""

    def __init__(
        self,
        game: str = "tictactoe",
        spool_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tracking_uri: str = "sqlite:///mlruns/mlflow.db",
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        resume: bool = True,
        in_channels: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        action_dim: Optional[int] = None,
        num_players: Optional[int] = None,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs_per_iter: int = 15,
        min_new_steps: int = 80,
        poll_interval: float = 1.5,
        eval_interval: int = 5,
        eval_games: int = 20,
        eval_baselines: Optional[list] = None,
        selfplay_bin: Optional[str] = None,
        tournament_bin: Optional[str] = None,
        selfplay_games: int = 50,
        selfplay_sims: Optional[int] = None,
        selfplay_cpuct: float = 1.414,
        selfplay_workers: int = 1,
        selfplay_parallel_games: int = 16,
        enable_selfplay: bool = True,
        steps_per_iter: Optional[int] = None,
        max_train_steps: Optional[int] = None,
        max_chunks_per_iter: Optional[int] = 1,
        max_spool_chunks: int = 4,
        updates_per_transition: Optional[float] = None,
        updates_per_trajectory: Optional[float] = None,
        replay_ratio: Optional[float] = None,
    ):
        self.game_cfg = get_game_config(game)
        self.game = self.game_cfg.name
        self.spool_dir = Path(spool_dir or f"./spool_{self.game}")
        self.model_path = Path(model_path or f"./models/{self.game}_latest.onnx")
        self.checkpoint_path = Path(checkpoint_path or f"./models/{self.game}_latest.pt")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or f"alphazero_{self.game}"
        self.run_name = run_name or f"run_{int(time.time())}"
        self.resume = resume

        self.in_channels = in_channels if in_channels is not None else self.game_cfg.in_channels
        self.height = height if height is not None else self.game_cfg.height
        self.width = width if width is not None else self.game_cfg.width
        self.action_dim = action_dim if action_dim is not None else self.game_cfg.action_dim
        self.num_players = num_players if num_players is not None else self.game_cfg.num_players
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.min_new_steps = min_new_steps
        self.poll_interval = poll_interval
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.eval_baselines = eval_baselines or list(self.game_cfg.eval_baselines)

        self.selfplay_bin = selfplay_bin or self.game_cfg.selfplay_bin
        self.tournament_bin = tournament_bin or self.game_cfg.tournament_bin
        self.selfplay_games = selfplay_games
        self.selfplay_sims = (
            selfplay_sims if selfplay_sims is not None else self.game_cfg.default_selfplay_sims
        )
        self.selfplay_cpuct = selfplay_cpuct
        self.selfplay_workers = max(1, selfplay_workers)
        self.selfplay_parallel_games = max(1, selfplay_parallel_games)
        self.enable_selfplay = enable_selfplay
        self.steps_per_iter = steps_per_iter
        self.max_train_steps = max(1, max_train_steps) if max_train_steps is not None else None
        self.max_chunks_per_iter = max_chunks_per_iter
        self.max_spool_chunks = max(1, max_spool_chunks)
        self.updates_per_transition = updates_per_transition
        self.updates_per_trajectory = updates_per_trajectory
        self.replay_ratio = replay_ratio

        self.stop_event = threading.Event()
        self.iteration = 0
        self.global_step = 0
        self.mlflow_run_id: Optional[str] = None

        # Build model and trainer
        self.model = AlphaZeroConvNet(
            in_channels=self.in_channels,
            height=self.height,
            width=self.width,
            action_dim=self.action_dim,
            num_players=self.num_players,
            hidden_channels=self.hidden_channels,
            num_res_blocks=self.num_res_blocks,
        )
        self.replay_buffer = ReplayBuffer(max_capacity=100_000)
        self.trainer = AlphaZeroTrainer(
            model=self.model,
            replay_buffer=self.replay_buffer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            in_channels=self.in_channels,
            height=self.height,
            width=self.width,
        )

        self.selfplay_supervisor = (
            SelfPlaySupervisor(
                binary_path=self.selfplay_bin,
                spool_dir=str(self.spool_dir),
                model_path=str(self.model_path),
                num_workers=self.selfplay_workers,
                games_per_batch=self.selfplay_games,
                sims=self.selfplay_sims,
                c_puct=self.selfplay_cpuct,
                parallel_games=self.selfplay_parallel_games,
                max_spool_chunks=self.max_spool_chunks,
            )
            if self.enable_selfplay
            else None
        )

        self.evaluator = TournamentEvaluator(binary_path=self.tournament_bin)

    def _setup_mlflow(self) -> None:
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Check if resuming an existing MLflow run
        if self.resume and self.mlflow_run_id:
            try:
                mlflow.start_run(run_id=self.mlflow_run_id)
                print(f"[MLflow] Resumed existing run '{self.mlflow_run_id}'")
                return
            except Exception as e:
                print(f"[MLflow] Could not resume run '{self.mlflow_run_id}': {e}. Starting new run.")

        mlflow.start_run(run_name=self.run_name)
        self.mlflow_run_id = mlflow.active_run().info.run_id
        print(f"[MLflow] Initialized active run '{self.run_name}' (ID: {self.mlflow_run_id})")

        # Log hyperparameters
        mlflow.log_params({
            "game": self.game,
            "in_channels": self.in_channels,
            "board_size": f"{self.height}x{self.width}",
            "action_dim": self.action_dim,
            "num_players": self.num_players,
            "hidden_channels": self.hidden_channels,
            "num_res_blocks": self.num_res_blocks,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs_per_iter": self.epochs_per_iter,
            "min_new_steps": self.min_new_steps,
            "selfplay_sims": self.selfplay_sims,
            "selfplay_cpuct": self.selfplay_cpuct,
            "selfplay_games_per_batch": self.selfplay_games,
            "selfplay_workers": self.selfplay_workers,
            "updates_per_transition": self.updates_per_transition,
            "updates_per_trajectory": self.updates_per_trajectory,
            "replay_ratio": self.replay_ratio,
        })

    def run(self, max_iterations: Optional[int] = None) -> None:
        """Executes the continuous actor-learner loop until max_iterations or Ctrl+C."""
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Restore checkpoint if resuming
        if self.checkpoint_path.exists():
            meta = self.trainer.load_checkpoint(self.checkpoint_path)
            self.iteration = meta.get("iteration", 0)
            self.global_step = meta.get("global_step", 0)
            self.mlflow_run_id = meta.get("mlflow_run_id", None)
            print(
                f"[Orchestrator] Restored checkpoint from '{self.checkpoint_path}' (Iter {self.iteration}, Step {self.global_step})"
            )

        # 2. Export baseline ONNX model if none exists
        if not self.model_path.exists():
            self.trainer.save_checkpoint(
                self.checkpoint_path,
                self.model_path,
                metadata={"iteration": self.iteration, "global_step": self.global_step},
            )
            print(f"[Orchestrator] Exported initial baseline model to '{self.model_path}'")

        # 3. Initialize MLflow
        self._setup_mlflow()

        # 4. Ingest any preexisting trajectory chunks
        initial_ingested = self.trainer.ingest(self.spool_dir, max_chunks=self.max_chunks_per_iter)
        if initial_ingested > 0:
            print(f"[Orchestrator] Pre-ingested {initial_ingested} existing steps into replay buffer.")

        # 5. Start self-play supervisor
        if self.selfplay_supervisor:
            self.selfplay_supervisor.start()

        # 6. Continuous Learning Loop
        print(f"\n{'='*70}\nStarting Continuous Training Loop (PID: {os.getpid()})\n{'='*70}")
        pending_steps = initial_ingested

        try:
            while not self.stop_event.is_set():
                if max_iterations is not None and self.iteration >= max_iterations:
                    print(f"[Orchestrator] Reached target max_iterations ({max_iterations}). Terminating.")
                    break

                # Sweep spool directory for newly written chunks
                newly_ingested = self.trainer.ingest(self.spool_dir, max_chunks=self.max_chunks_per_iter)
                pending_steps += newly_ingested

                # If enough new samples or initial bootstrap buffer is ready
                if pending_steps >= self.min_new_steps or (
                    len(self.replay_buffer) >= 64 and self.iteration == 0
                ):
                    self.iteration += 1
                    t0 = time.time()

                    # Execute training updates (fixed steps, per-transition/trajectory/replay-ratio, or capped epoch-based)
                    if self.steps_per_iter is not None:
                        target_steps = self.steps_per_iter
                    elif self.updates_per_transition is not None:
                        target_steps = max(1, round(pending_steps * self.updates_per_transition))
                    elif self.updates_per_trajectory is not None:
                        est_trajs = max(1.0, pending_steps / 7.5)
                        target_steps = max(1, round(est_trajs * self.updates_per_trajectory))
                    elif self.replay_ratio is not None:
                        target_steps = max(
                            1, round((pending_steps * self.replay_ratio) / self.batch_size)
                        )
                    else:
                        target_steps = max(
                            1, (len(self.replay_buffer) // self.batch_size) * self.epochs_per_iter
                        )

                    if self.max_train_steps is not None:
                        target_steps = max(1, min(target_steps, self.max_train_steps))

                    tot_loss, p_loss, v_loss, steps = self.trainer.train_steps(
                        num_steps=target_steps,
                        batch_size=self.batch_size,
                    )
                    self.global_step += steps
                    elapsed = time.time() - t0

                    # Save updated ONNX model and checkpoint
                    meta = {
                        "iteration": self.iteration,
                        "global_step": self.global_step,
                        "mlflow_run_id": self.mlflow_run_id,
                    }
                    self.trainer.save_checkpoint(
                        checkpoint_path=self.checkpoint_path,
                        onnx_out_path=self.model_path,
                        metadata=meta,
                    )

                    # Log metrics to MLflow
                    metrics = {
                        "loss/total": tot_loss,
                        "loss/policy": p_loss,
                        "loss/value": v_loss,
                        "buffer/total_steps": len(self.replay_buffer),
                        "buffer/ingested_steps": pending_steps,
                        "train/iteration_time_sec": elapsed,
                        "iteration": self.iteration,
                    }
                    mlflow.log_metrics(metrics, step=self.global_step)

                    print(
                        f"[Iter {self.iteration:4d}] Loss = {tot_loss:.4f} (P = {p_loss:.4f}, V = {v_loss:.4f}) | "
                        f"Buffer = {len(self.replay_buffer):5d} (+{pending_steps:3d}) | "
                        f"Step = {self.global_step:5d} ({elapsed:.1f}s)"
                    )
                    if self.updates_per_transition is not None and self.max_train_steps is not None:
                        consumed = round(steps / self.updates_per_transition)
                        pending_steps = max(0, pending_steps - consumed)
                    elif self.replay_ratio is not None and self.max_train_steps is not None:
                        consumed = round((steps * self.batch_size) / self.replay_ratio)
                        pending_steps = max(0, pending_steps - consumed)
                    else:
                        pending_steps = 0

                    # Periodic Tournament Evaluation
                    if self.eval_interval > 0 and self.iteration % self.eval_interval == 0:
                        print(f"[Orchestrator] Running tournament evaluation at Iteration {self.iteration}...")
                        eval_metrics = self.evaluator.evaluate(
                            model_path=str(self.model_path),
                            num_sims=self.selfplay_sims,
                            games=self.eval_games,
                            baselines=self.eval_baselines,
                        )
                        if eval_metrics:
                            mlflow.log_metrics(eval_metrics, step=self.global_step)
                            summary_parts = []
                            for b in self.eval_baselines:
                                w = eval_metrics.get(f"eval/{b}_win_rate", 0.0) * 100
                                d = eval_metrics.get(f"eval/{b}_draw_rate", 0.0) * 100
                                l = eval_metrics.get(f"eval/{b}_loss_rate", 0.0) * 100
                                summary_parts.append(f"vs {b.capitalize()}: {w:.0f}% W / {d:.0f}% D / {l:.0f}% L")
                            print(f"  --> Eval Standings: {' | '.join(summary_parts)}")

                else:
                    time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n[Orchestrator] KeyboardInterrupt received. Initiating graceful shutdown...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shuts down background workers and finalizes MLflow run."""
        self.stop_event.set()
        if self.selfplay_supervisor:
            self.selfplay_supervisor.stop()

        # Save final model state
        meta = {
            "iteration": self.iteration,
            "global_step": self.global_step,
            "mlflow_run_id": self.mlflow_run_id,
        }
        self.trainer.save_checkpoint(
            checkpoint_path=self.checkpoint_path,
            onnx_out_path=self.model_path,
            metadata=meta,
        )

        # Log final artifacts to MLflow
        try:
            if self.model_path.exists():
                mlflow.log_artifact(str(self.model_path), artifact_path="models")
            meta_json = self.checkpoint_path.with_suffix(".meta.json")
            if meta_json.exists():
                mlflow.log_artifact(str(meta_json), artifact_path="checkpoints")
            mlflow.end_run()
            print(f"[Orchestrator] Finalized MLflow run '{self.mlflow_run_id}'.")
        except Exception as e:
            print(f"[Orchestrator] Error finalizing MLflow run: {e}")

        print("[Orchestrator] Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous AlphaZero Actor-Learner Experiment Orchestrator with MLflow tracking"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="tictactoe",
        choices=["tictactoe", "connect4"],
        help="Target game environment (default: tictactoe)",
    )
    parser.add_argument("--spool-dir", type=str, default=None, help="Directory holding binary trajectory chunks")
    parser.add_argument("--model-path", type=str, default=None, help="Output ONNX model path")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="PyTorch checkpoint path")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlruns/mlflow.db")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh run instead of resuming")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N training iterations")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--min-new-steps", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=5, help="Tournament evaluation interval")
    parser.add_argument("--eval-games", type=int, default=20, help="Games per evaluation tournament")
    parser.add_argument("--selfplay-games", type=int, default=50, help="Games per self-play batch")
    parser.add_argument("--selfplay-sims", type=int, default=None, help="MCTS simulations in self-play")
    parser.add_argument(
        "--selfplay-workers",
        type=int,
        default=1,
        help="Number of parallel self-play worker processes to run",
    )
    parser.add_argument(
        "--selfplay-parallel-games",
        type=int,
        default=16,
        help="Number of concurrent game trees managed in lockstep by MultiGameScheduler",
    )
    parser.add_argument("--no-selfplay", action="store_true", help="Do not spawn self-play worker subprocess")
    parser.add_argument(
        "--updates-per-transition",
        "--steps-per-sample",
        type=float,
        default=None,
        dest="updates_per_transition",
        help="Gradient weight updates per newly ingested transition/sample (e.g. 0.25)",
    )
    parser.add_argument(
        "--updates-per-trajectory",
        type=float,
        default=None,
        help="Gradient weight updates per newly completed game trajectory (e.g. 2.0)",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=None,
        help="Data reuse ratio: average times each transition is sampled (e.g. 8.0)",
    )
    parser.add_argument(
        "--steps-per-iter",
        type=int,
        default=None,
        help="Fixed number of mini-batch gradient updates per iteration (e.g. 50 or 100)",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Optional safety cap on the maximum gradient steps executed in a single iteration (default: None)",
    )
    parser.add_argument(
        "--max-chunks-per-iter",
        type=int,
        default=1,
        help="Maximum spool chunks ingested per iteration (default: 1)",
    )
    parser.add_argument(
        "--max-spool-chunks",
        type=int,
        default=4,
        help="Maximum unconsumed chunks in spool directory before pausing workers (default: 4)",
    )

    args = parser.parse_args()

    orchestrator = ExperimentOrchestrator(
        game=args.game,
        spool_dir=args.spool_dir,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        resume=not args.no_resume,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs_per_iter=args.epochs,
        min_new_steps=args.min_new_steps,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        selfplay_workers=args.selfplay_workers,
        selfplay_parallel_games=args.selfplay_parallel_games,
        enable_selfplay=not args.no_selfplay,
        steps_per_iter=args.steps_per_iter,
        max_train_steps=args.max_train_steps,
        max_chunks_per_iter=args.max_chunks_per_iter,
        max_spool_chunks=args.max_spool_chunks,
        updates_per_transition=args.updates_per_transition,
        updates_per_trajectory=args.updates_per_trajectory,
        replay_ratio=args.replay_ratio,
    )

    # Register OS signal handlers for clean interrupt handling
    def _sig_handler(signum, frame):
        print(f"\nSignal {signum} received. Stopping orchestrator...")
        orchestrator.stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    orchestrator.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
