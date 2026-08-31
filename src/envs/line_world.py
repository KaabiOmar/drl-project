"""Line World : environnement de test, 1 dimension.

Le pion demarre au centre d'un couloir de `size` cases. Atteindre la case la
plus a droite rapporte +1, la case la plus a gauche -1. Les deux extremites
sont terminales.

Sert de sanity check : tout agent correct doit converger vers un score moyen
proche de +1 en quelques milliers d'episodes.
"""

from __future__ import annotations

import numpy as np

from src.envs.base import Env

LEFT, RIGHT = 0, 1


class LineWorld(Env):
    name = "line_world"

    def __init__(self, size: int = 5, max_steps: int = 100) -> None:
        if size < 3:
            raise ValueError("size doit valoir au moins 3")
        self.size = size
        self.max_steps = max_steps
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.size

    @property
    def num_actions(self) -> int:
        return 2

    def reset(self) -> None:
        self.position = self.size // 2
        self._score = 0.0
        self._over = False
        self.steps = 0

    def state_description(self) -> np.ndarray:
        state = np.zeros(self.size, dtype=np.float32)
        state[self.position] = 1.0
        return state

    def available_actions_mask(self) -> np.ndarray:
        # Les deux actions restent toujours legales : le mur est absorbant.
        return np.ones(2, dtype=np.float32)

    def step(self, action: int) -> None:
        if self._over:
            raise RuntimeError("step() appele sur un episode termine")
        self.position += 1 if action == RIGHT else -1
        self.steps += 1
        if self.position == 0:
            self._score -= 1.0
            self._over = True
        elif self.position == self.size - 1:
            self._score += 1.0
            self._over = True
        elif self.steps >= self.max_steps:
            self._over = True

    def is_game_over(self) -> bool:
        return self._over

    def score(self) -> float:
        return self._score

    def clone(self) -> "LineWorld":
        copy = LineWorld(self.size, self.max_steps)
        copy.position = self.position
        copy._score = self._score
        copy._over = self._over
        copy.steps = self.steps
        return copy

    def render(self) -> str:
        cells = ["." for _ in range(self.size)]
        cells[0], cells[-1] = "-", "+"
        cells[self.position] = "X"
        return "".join(cells)
