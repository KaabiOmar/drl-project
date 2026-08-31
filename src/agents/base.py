"""Contrat commun a tous les agents.

Un agent ne connait jamais l'environnement concret : il recoit un vecteur
d'etat et un masque d'actions. Les agents de planification (RandomRollout, MCTS)
font exception et recoivent `env` pour pouvoir simuler via `env.clone()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.common.replay_buffer import Transition


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        env=None,
        greedy: bool = False,
    ) -> int:
        """Choisit une action legale.

        `greedy=True` signifie POLITIQUE OBTENUE : aucune exploration, c'est le
        mode impose pour toutes les metriques du rapport.
        """

    def observe(self, transition: Transition) -> None:
        """Recoit une transition. Sans effet pour les agents non apprenants."""

    def end_episode(self) -> None:
        """Fin d'episode : moment de la mise a jour pour REINFORCE et PPO."""

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modele. Livrable impose : les modeles doivent etre rejouables."""

    def load(self, path: str | Path) -> None:
        """Recharge un modele sauvegarde."""
