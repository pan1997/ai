"""Autonomous training service for AlphaZero dual-headed ConvNet from spooled chunks."""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from learner.models.convnet import AlphaZeroConvNet, export_alphazero_onnx
from learner.trainer import ReplayBuffer, compute_alphazero_loss, ingest_spool_chunks


def train_iteration(
    spool_dir: Path | str,
    onnx_out_path: Path | str,
    replay_buffer: ReplayBuffer,
    in_channels: int = 3,
    height: int = 3,
    width: int = 3,
    action_dim: int = 9,
    num_players: int = 2,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.002,
    weight_decay: float = 1e-4,
    checkpoint_path: Path | str | None = None,
) -> AlphaZeroConvNet:
    model = AlphaZeroConvNet(
        in_channels=in_channels,
        height=height,
        width=width,
        action_dim=action_dim,
        num_players=num_players,
        hidden_channels=64,
        num_res_blocks=2,
    )

    ckpt_file = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_file and ckpt_file.exists():
        model.load_state_dict(torch.load(ckpt_file, weights_only=True))
        print(f"Loaded existing PyTorch checkpoint from {ckpt_file}")

    ingested = ingest_spool_chunks(spool_dir, replay_buffer)
    print(f"Ingested {ingested} new steps from {spool_dir}. Buffer capacity: {len(replay_buffer)}")

    if len(replay_buffer) < 10:
        print("Not enough samples in replay buffer to train. Skipping gradient update.")
        export_alphazero_onnx(model, onnx_out_path, (in_channels, height, width))
        return model

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()

    actual_batch_size = min(batch_size, len(replay_buffer))
    steps_per_epoch = max(1, len(replay_buffer) // actual_batch_size)

    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            batch = replay_buffer.sample_batch(actual_batch_size)
            obs_batch = np.stack([r["obs"] for r in batch]).reshape(-1, in_channels, height, width)
            policy_batch = np.stack([r["policy"] for r in batch])
            value_batch = np.stack([r["value"] for r in batch])

            x = torch.from_numpy(obs_batch).float()
            target_p = torch.from_numpy(policy_batch).float()
            target_v = torch.from_numpy(value_batch).float()

            optimizer.zero_grad()
            p_logits, v_pred = model(x)
            total_loss, p_loss, v_loss = compute_alphazero_loss(p_logits, v_pred, target_p, target_v)
            total_loss.backward()
            optimizer.step()

            epoch_losses.append((total_loss.item(), p_loss.item(), v_loss.item()))

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            avg_tot = np.mean([l[0] for l in epoch_losses])
            avg_p = np.mean([l[1] for l in epoch_losses])
            avg_v = np.mean([l[2] for l in epoch_losses])
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_tot:.4f} (Policy = {avg_p:.4f}, Value = {avg_v:.4f})")

    # Export ONNX model for MCTS search workers
    export_alphazero_onnx(model, onnx_out_path, (in_channels, height, width))
    print(f"Exported updated ONNX model to {onnx_out_path}")

    if ckpt_file:
        torch.save(model.state_dict(), ckpt_file)
        print(f"Saved PyTorch weights to {ckpt_file}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero ConvNet from trajectory chunks")
    parser.add_argument("--spool-dir", type=str, default="./spool_tictactoe")
    parser.add_argument("--model-path", type=str, default="./models/latest.onnx")
    parser.add_argument("--checkpoint-path", type=str, default="./models/latest.pt")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=9)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    buffer = ReplayBuffer(max_capacity=100_000)
    train_iteration(
        spool_dir=args.spool_dir,
        onnx_out_path=args.model_path,
        replay_buffer=buffer,
        in_channels=args.channels,
        height=args.height,
        width=args.width,
        action_dim=args.action_dim,
        num_players=args.players,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
