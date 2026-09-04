"""TicTacToe contre un adversaire aleatoire.

L'agent joue toujours les croix et commence. L'adversaire joue a l'interieur de
`step()`, immediatement apres le coup de l'agent : vu de l'agent,
l'environnement reste un MDP a un joueur, simplement stochastique.

Encodage de l'etat : 3 plans de 9 cases (vide / agent / adversaire) = 27.
Encodage de l'action : identifiant de la case jouee, 0..8.
"""

from __future__ import annotations

import copy

import numpy as np

from src.envs.base import Env

EMPTY, AGENT, OPPONENT = 0, 1, 2
_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # lignes
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colonnes
    (0, 4, 8), (2, 4, 6),             # diagonales
)


class TicTacToe(Env):
    name = "tictactoe"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        # Politique adverse injectable. None = tirage uniforme, le comportement
        # par defaut et celui de toutes nos mesures. L'interface graphique s'en
        # sert pour faire jouer l'adversaire par un modele entraine.
        self.external_opponent = None
        self.reset()

    @property
    def state_dim(self) -> int:
        return 27

    @property
    def num_actions(self) -> int:
        return 9

    def reset(self) -> None:
        self.board = np.zeros(9, dtype=np.int8)
        self._score = 0.0
        self._over = False

    def state_description(self) -> np.ndarray:
        state = np.zeros(27, dtype=np.float32)
        for cell in range(9):
            state[self.board[cell] * 9 + cell] = 1.0
        return state

    def available_actions_mask(self) -> np.ndarray:
        return (self.board == EMPTY).astype(np.float32)

    def step(self, action: int) -> None:
        if self._over:
            raise RuntimeError("step() appele sur un episode termine")
        action = int(action)
        if self.board[action] != EMPTY:
            raise ValueError(f"action illegale: case {action} deja occupee")

        self.board[action] = AGENT
        if self._settle(AGENT):
            return

        empty = np.flatnonzero(self.board == EMPTY)
        if self.external_opponent is not None:
            choix = int(self.external_opponent(self, empty))
        else:
            choix = int(self.rng.choice(empty))
        self.board[choix] = OPPONENT
        self._settle(OPPONENT)

    def _settle(self, player: int) -> bool:
        """Met a jour score et terminaison apres le coup de `player`."""
        if self._has_won(player):
            self._score += 1.0 if player == AGENT else -1.0
            self._over = True
        elif not (self.board == EMPTY).any():
            self._over = True  # match nul, score inchange
        return self._over

    def _has_won(self, player: int) -> bool:
        return any(all(self.board[cell] == player for cell in line) for line in _LINES)

    def is_game_over(self) -> bool:
        return self._over

    def score(self) -> float:
        return self._score

    def clone(self) -> "TicTacToe":
        new = TicTacToe.__new__(TicTacToe)
        new.external_opponent = self.external_opponent
        new.rng = copy.deepcopy(self.rng)
        new.board = self.board.copy()
        new._score = self._score
        new._over = self._over
        return new

    def render(self) -> str:
        glyphs = {EMPTY: ".", AGENT: "X", OPPONENT: "O"}
        rows = ["".join(glyphs[int(self.board[r * 3 + c])] for c in range(3)) for r in range(3)]
        return "\n".join(rows)
