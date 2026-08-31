"""Agent aleatoire uniforme sur les actions legales.

C'est la reference basse de toutes les comparaisons : un algorithme qui ne le
bat pas n'a rien appris.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None, **_: object) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state, mask, env=None, greedy: bool = False) -> int:
        legal = np.flatnonzero(mask > 0.0)
        return int(self.rng.choice(legal))
