"""Grid World : environnement de test, 2 dimensions.

Grille `rows x cols`, depart en haut a gauche. Le coin bas-droite est un but
(+1), le coin haut-droite un piege (-1). Les deux sont terminaux, les murs sont
absorbants (une action qui sort de la grille laisse l'agent sur place).
"""

from __future__ import annotations

import numpy as np

from src.envs.base import Env

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class GridWorld(Env):
    name = "grid_world"

    def __init__(self, rows: int = 5, cols: int = 5, max_steps: int = 200) -> None:
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.goal = (rows - 1, cols - 1)
        self.trap = (0, cols - 1)
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.rows * self.cols

    @property
    def num_actions(self) -> int:
        return 4

    def reset(self) -> None:
        self.row, self.col = 0, 0
        self._score = 0.0
        self._over = False
        self.steps = 0

    def state_description(self) -> np.ndarray:
        state = np.zeros(self.rows * self.cols, dtype=np.float32)
        state[self.row * self.cols + self.col] = 1.0
        return state

    def available_actions_mask(self) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def step(self, action: int) -> None:
        if self._over:
            raise RuntimeError("step() appele sur un episode termine")
        d_row, d_col = _DELTAS[int(action)]
        self.row = min(max(self.row + d_row, 0), self.rows - 1)
        self.col = min(max(self.col + d_col, 0), self.cols - 1)
        self.steps += 1
        if (self.row, self.col) == self.goal:
            self._score += 1.0
            self._over = True
        elif (self.row, self.col) == self.trap:
            self._score -= 1.0
            self._over = True
        elif self.steps >= self.max_steps:
            self._over = True

    def is_game_over(self) -> bool:
        return self._over

    def score(self) -> float:
        return self._score

    def clone(self) -> "GridWorld":
        copy = GridWorld(self.rows, self.cols, self.max_steps)
        copy.row, copy.col = self.row, self.col
        copy._score = self._score
        copy._over = self._over
        copy.steps = self.steps
        return copy

    def render(self) -> str:
        lines = []
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                if (r, c) == (self.row, self.col):
                    line += "X"
                elif (r, c) == self.goal:
                    line += "+"
                elif (r, c) == self.trap:
                    line += "-"
                else:
                    line += "."
            lines.append(line)
        return "\n".join(lines)
