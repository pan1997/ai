"""Training loop for TabularAlphaZero on discrete GraphEnv MDP."""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from learner.models.tabular import TabularAlphaZero
from learner.spool_reader import read_trajectory_chunk
from learner.trainer import ReplayBuffer, compute_alphazero_loss, ingest_spool_chunks


def train_tabular_iteration(
    spool_dir: Path | str,
    model_path: Path | str,
    replay_buffer: ReplayBuffer,
    num_states: int = 4,
    num_actions: int = 2,
    num_players: int = 1,
    learning_rate: float = 0.2,
    train_epochs: int = 50,
) -> TabularAlphaZero:
    model_file = Path(model_path)
    if model_file.exists():
        model = TabularAlphaZero.load_binary(model_file)
    else:
        model = TabularAlphaZero(num_states, num_actions, num_players)

    # Ingest new trajectory chunks from spool directory
    ingested = ingest_spool_chunks(spool_dir, replay_buffer)
    print(f"Ingested {ingested} steps from {spool_dir}. Buffer size: {len(replay_buffer)}")

    if len(replay_buffer) == 0:
        return model

    # Convert replay buffer records into PyTorch training tensors
    obs_list = [r["obs"] for r in replay_buffer.buffer]
    policy_list = [r["policy"] for r in replay_buffer.buffer]
    value_list = [r["value"] for r in replay_buffer.buffer]

    obs_np = np.stack(obs_list)
    state_ids = torch.from_numpy(obs_np.argmax(axis=-1).astype(np.int64))
    target_policy = torch.from_numpy(np.stack(policy_list).astype(np.float32))
    target_value = torch.from_numpy(np.stack(value_list).astype(np.float32))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(train_epochs):
        optimizer.zero_grad()
        pred_logits, pred_values = model(state_ids)
        total_loss, p_loss, v_loss = compute_alphazero_loss(
            pred_logits, pred_values, target_policy, target_value
        )
        total_loss.backward()
        optimizer.step()

    # Save updated weights atomically
    model.save_binary(model_file)
    print(f"Saved updated tabular model to {model_file}")

    # Print summary table
    priors = torch.softmax(model.policy_table, dim=-1).detach().numpy()
    values = model.value_table.detach().numpy()
    print("--- Tabular Model Status ---")
    for s in range(num_states):
        p_str = ", ".join(f"{p:.3f}" for p in priors[s])
        v_str = ", ".join(f"{v:.3f}" for v in values[s])
        print(f"State {s}: Priors = [{p_str}], Value = [{v_str}]")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Tabular AlphaZero on spooled chunks")
    parser.add_argument("--spool-dir", type=str, default="./spool_graph")
    parser.add_argument("--model-path", type=str, default="./tabular_model.bin")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--actions", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.2)
    args = parser.parse_args()

    buffer = ReplayBuffer(max_capacity=50_000)
    train_tabular_iteration(
        args.spool_dir,
        args.model_path,
        buffer,
        num_states=args.states,
        num_actions=args.actions,
        train_epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
