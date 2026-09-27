"""AlphaZero dual-headed convolutional neural network."""

import os
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual convolutional block with batch normalization and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class AlphaZeroConvNet(nn.Module):
    """AlphaZero dual-headed neural network evaluating policy logits and value estimates."""

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        action_dim: int,
        num_players: int = 2,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.num_players = num_players

        # Input feature projection
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Residual backbone
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(num_res_blocks)])

        # Dedicated Policy Heads (One per player)
        self.policy_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_players)
        ])
        self.policy_fcs = nn.ModuleList([
            nn.Linear(2 * height * width, action_dim)
            for _ in range(num_players)
        ])

        # Dedicated Value Heads (One per player, each outputting scalar in [-1, +1])
        self.value_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_players)
        ])
        self.value_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 * height * width, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Tanh(),
            )
            for _ in range(num_players)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing active policy logits and multi-agent value estimates.

        Args:
            x: Input observation tensor of shape `(batch_size, in_channels, height, width)`.

        Returns:
            policy_logits: `(batch_size, action_dim)` for active player.
            values: `(batch_size, num_players)` where index `p` is the expected return for player `p`.
        """
        out = self.stem(x)
        for block in self.res_blocks:
            out = block(out)

        # 1. Multi-Agent Value Vector: evaluated by each player's dedicated value head
        v_list = []
        for i in range(self.num_players):
            v = self.value_convs[i](out)
            v = torch.flatten(v, start_dim=1)
            v = self.value_fcs[i](v)
            v_list.append(v)
        values = torch.cat(v_list, dim=-1)

        # 2. Player Policy Heads
        p_list = []
        for i in range(self.num_players):
            p = self.policy_convs[i](out)
            p = torch.flatten(p, start_dim=1)
            p = self.policy_fcs[i](p)
            p_list.append(p)

        if self.num_players == 1:
            policy_logits = p_list[0]
        elif self.in_channels > self.num_players and self.num_players == 2:
            # Active player turn indicator at channel index self.num_players (e.g. channel 2)
            # 1.0 = Player 0 (X), 0.0 = Player 1 (O)
            turn = x[:, self.num_players : self.num_players + 1, 0, 0]
            policy_logits = turn * p_list[0] + (1.0 - turn) * p_list[1]
        elif self.in_channels >= self.num_players * 2:
            # General N-player one-hot active channels at indices num_players..2*num_players
            stacked = torch.stack(p_list, dim=1)
            indicators = x[:, self.num_players : self.num_players + self.num_players, 0, 0].unsqueeze(-1)
            policy_logits = (stacked * indicators).sum(dim=1)
        else:
            policy_logits = p_list[0]

        return policy_logits, values

    def forward_all_heads(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes policy logits for ALL players alongside value estimates.

        Returns:
            all_policies: `(batch_size, num_players, action_dim)`
            values: `(batch_size, num_players)`
        """
        out = self.stem(x)
        for block in self.res_blocks:
            out = block(out)

        v_list = [
            self.value_fcs[i](torch.flatten(self.value_convs[i](out), start_dim=1))
            for i in range(self.num_players)
        ]
        values = torch.cat(v_list, dim=-1)

        p_list = [
            self.policy_fcs[i](torch.flatten(self.policy_convs[i](out), start_dim=1))
            for i in range(self.num_players)
        ]
        all_policies = torch.stack(p_list, dim=1)

        return all_policies, values


def export_alphazero_onnx(
    model: nn.Module,
    save_path: Path | str,
    input_shape: Tuple[int, int, int],
) -> None:
    """Exports a PyTorch model to ONNX format with dynamic batch axes and atomic replacement.

    Args:
        model: PyTorch AlphaZero neural network.
        save_path: Destination path for the `.onnx` model file.
        input_shape: Tuple of `(channels, height, width)`.
    """
    path = Path(save_path)
    tmp_path = path.with_suffix(".onnx.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(1, *input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(tmp_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        dynamo=False,
    )

    os.replace(tmp_path, path)
