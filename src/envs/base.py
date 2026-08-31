"""Contrat commun a tous les environnements du projet.

Cette interface est FIGEE : les agents ne doivent jamais importer un
environnement concret, uniquement manipuler `Env`. Toute methode ajoutee ici
doit l'etre dans les quatre environnements en meme temps.

Conventions
-----------
* `state_description()` renvoie toujours un vecteur `float32` de taille fixe
  `state_dim`, du point de vue du joueur controle par l'agent.
* L'espace d'actions est GLOBAL et de taille fixe `num_actions` ; la legalite
  est portee par `available_actions_mask()`, jamais par un changement de taille.
* `score()` est le score CUMULE de l'episode en cours (et non la recompense du
  dernier pas). La recompense d'un pas se calcule par difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Env(ABC):
    """Environnement sequentiel a un seul agent apprenant.

    Les environnements a deux joueurs (TicTacToe, Bobail) integrent l'adversaire
    a l'interieur de `step()` : du point de vue de l'agent, l'environnement
    reste un MDP a un joueur.
    """

    name: str = "env"

    # --- description statique -------------------------------------------------

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Taille du vecteur d'etat."""

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Taille de l'espace d'actions global."""

    # --- cycle de vie ---------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Remet l'environnement dans son etat initial."""

    @abstractmethod
    def state_description(self) -> np.ndarray:
        """Vecteur d'etat `float32` de forme `(state_dim,)`."""

    @abstractmethod
    def available_actions_mask(self) -> np.ndarray:
        """Masque `float32` de forme `(num_actions,)` : 1.0 si l'action est legale."""

    @abstractmethod
    def step(self, action: int) -> None:
        """Applique une action legale et met a jour score / terminaison."""

    @abstractmethod
    def is_game_over(self) -> bool:
        """Vrai si l'episode est termine."""

    @abstractmethod
    def score(self) -> float:
        """Score cumule de l'episode en cours."""

    @abstractmethod
    def clone(self) -> "Env":
        """Copie profonde et independante, utilisee par RandomRollout et MCTS.

        La copie ne doit partager AUCUN etat mutable avec l'original (plateau,
        compteurs, generateur aleatoire).
        """

    # --- helpers derives ------------------------------------------------------

    def available_actions(self) -> np.ndarray:
        """Identifiants des actions legales."""
        return np.flatnonzero(self.available_actions_mask() > 0.0)

    def render(self) -> str:
        """Rendu texte, pour le debug et les tests."""
        return f"<{self.name} score={self.score()} over={self.is_game_over()}>"
