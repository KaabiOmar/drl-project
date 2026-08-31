"""REINFORCE avec baseline constante (moyenne des retours).

Delta par rapport a `ReinforceAgent` : soustraire aux retours `G_t` leur moyenne
(sur l'episode, ou une moyenne glissante sur les episodes recents) avant de
calculer le gradient.

Ce que cela change et qu'il faut savoir demontrer : la baseline ne modifie pas
l'esperance du gradient (elle est non biaisee) mais en reduit la variance. La
comparaison des courbes avec et sans baseline est un resultat attendu.
"""

from __future__ import annotations

from src.agents.reinforce import ReinforceAgent


class ReinforceMeanBaselineAgent(ReinforceAgent):
    name = "reinforce_baseline"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "ReinforceMeanBaselineAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
