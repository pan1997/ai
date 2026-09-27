"""Game configuration definitions and registry for AlphaZero experiments."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class GameConfig:
    """Configuration specification for a game environment."""

    name: str
    in_channels: int
    height: int
    width: int
    action_dim: int
    num_players: int = 2
    selfplay_bin: str = ""
    tournament_bin: str = ""
    eval_baselines: List[str] = field(default_factory=lambda: ["tactical", "random"])
    default_selfplay_sims: int = 40
    default_eval_sims: int = 50


GAME_REGISTRY = {
    "tictactoe": GameConfig(
        name="tictactoe",
        in_channels=3,
        height=3,
        width=3,
        action_dim=9,
        num_players=2,
        selfplay_bin="target/release/tictactoe-selfplay",
        tournament_bin="target/release/tictactoe-tournament",
        eval_baselines=["tactical", "random"],
        default_selfplay_sims=40,
        default_eval_sims=50,
    ),
    "connect4": GameConfig(
        name="connect4",
        in_channels=3,
        height=6,
        width=7,
        action_dim=7,
        num_players=2,
        selfplay_bin="target/release/connect4-selfplay",
        tournament_bin="target/release/connect4-tournament",
        eval_baselines=["tactical", "random"],
        default_selfplay_sims=50,
        default_eval_sims=50,
    ),
}


def get_game_config(name: str) -> GameConfig:
    """Retrieves a GameConfig by name from the registry, raising ValueError if not found."""
    canonical = name.strip().lower()
    if canonical not in GAME_REGISTRY:
        supported = ", ".join(sorted(GAME_REGISTRY.keys()))
        raise ValueError(f"Unknown game '{name}'. Supported games: {supported}")
    return GAME_REGISTRY[canonical]
