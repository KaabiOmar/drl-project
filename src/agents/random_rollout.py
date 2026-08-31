"""Random Rollout : evaluation d'un coup par simulations aleatoires.

Pour chaque action legale, on clone l'environnement, on joue le coup, puis on
termine la partie au hasard `rollouts_per_action` fois. On garde l'action dont
le score final moyen est le meilleur.

Agent de planification : aucun apprentissage, aucun modele a sauvegarder. Son
interet dans le rapport est double, comme plancher de qualite pour MCTS et comme
illustration du cout en TEMPS PAR COUP, metrique explicitement demandee.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import Agent


class RandomRolloutAgent(Agent):
    name = "random_rollout"

    def __init__(self, rollouts_per_action: int = 20, seed: int | None = None, **_: object) -> None:
        self.rollouts_per_action = rollouts_per_action
        self.rng = np.random.default_rng(seed)

    def act(self, state, mask, env=None, greedy: bool = False) -> int:
        if env is None:
            raise ValueError("RandomRolloutAgent a besoin de `env` pour simuler")
        legal = np.flatnonzero(mask > 0.0)
        best_action, best_value = int(legal[0]), -np.inf
        for action in legal:
            value = np.mean([self._rollout(env, int(action)) for _ in range(self.rollouts_per_action)])
            if value > best_value:
                best_action, best_value = int(action), value
        return best_action

    def _rollout(self, env, action: int) -> float:
        simulation = env.clone()
        simulation.step(action)
        while not simulation.is_game_over():
            legal = np.flatnonzero(simulation.available_actions_mask() > 0.0)
            if legal.size == 0:
                break
            simulation.step(int(self.rng.choice(legal)))
        return simulation.score()
