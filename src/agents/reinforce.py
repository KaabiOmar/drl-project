"""REINFORCE (Monte Carlo policy gradient).

A implementer :
* un reseau de politique `mlp(state_dim, num_actions)` dont les logits sont
  MASQUES (`apply_mask_torch`) avant le softmax ;
* `act()` echantillonne dans la distribution ; en mode `greedy=True` on prend
  l'argmax masque, c'est la politique obtenue ;
* memoriser (log_prob, recompense) de tout l'episode, puis a `end_episode()`
  calculer les retours actualises `G_t` et remonter le gradient
  `- sum_t log pi(a_t|s_t) * G_t`.

Aucune mise a jour en cours d'episode : c'est une methode Monte Carlo, ce qui
explique sa variance elevee - point central du rapport.
"""

from __future__ import annotations

from src.agents.base import Agent


class ReinforceAgent(Agent):
    name = "reinforce"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "ReinforceAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
