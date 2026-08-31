"""Environnements du projet, exposes par un registre unique."""

from __future__ import annotations

from typing import Callable

from src.envs.base import Env
from src.envs.bobail import Bobail
from src.envs.grid_world import GridWorld
from src.envs.line_world import LineWorld
from src.envs.tictactoe import TicTacToe

ENV_REGISTRY: dict[str, Callable[..., Env]] = {
    "line_world": LineWorld,
    "grid_world": GridWorld,
    "tictactoe": TicTacToe,
    "bobail": Bobail,
}


def make_env(name: str, **kwargs) -> Env:
    """Instancie un environnement par son nom de registre."""
    if name not in ENV_REGISTRY:
        raise KeyError(f"environnement inconnu: {name!r} (connus: {sorted(ENV_REGISTRY)})")
    return ENV_REGISTRY[name](**kwargs)


__all__ = ["Env", "LineWorld", "GridWorld", "TicTacToe", "Bobail", "ENV_REGISTRY", "make_env"]
