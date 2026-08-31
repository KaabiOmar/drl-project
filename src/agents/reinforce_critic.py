"""REINFORCE avec baseline apprise par un critique.

Delta : la baseline devient un reseau de valeur `V(s)` entraine en parallele par
regression sur les retours observes. L'avantage devient `G_t - V(s_t)`.

Deux pertes distinctes a suivre separement dans les logs : la perte de politique
et la perte du critique. Ne pas retropropager la perte du critique dans la
politique (`.detach()` sur l'avantage).
"""

from __future__ import annotations

from src.agents.reinforce import ReinforceAgent


class ReinforceCriticBaselineAgent(ReinforceAgent):
    name = "reinforce_critic"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "ReinforceCriticBaselineAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
