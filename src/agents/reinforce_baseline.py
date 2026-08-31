"""REINFORCE avec baseline constante : la moyenne des retours de l'episode.

Un seul changement : on soustrait aux retours leur moyenne avant de calculer le
gradient.

Pourquoi c'est legitime, et c'est LE point a savoir demontrer : soustraire une
quantite `b` qui ne depend pas de l'action ne change pas l'esperance du
gradient, parce que `E[grad log pi(a|s)] = 0`. Le gradient reste donc non
biaise, mais sa variance baisse.

Pourquoi ca aide concretement : sans baseline, si tous les retours d'un episode
sont positifs, toutes les actions voient leur probabilite augmenter - y compris
les mauvaises, juste un peu moins que les bonnes. Avec la baseline, les actions
sous la moyenne sont explicitement decouragees.

La comparaison des courbes avec et sans baseline est un resultat attendu du
rapport : meme score final, moins d'oscillations.
"""

from __future__ import annotations

from src.agents.reinforce import ReinforceAgent


class ReinforceMeanBaselineAgent(ReinforceAgent):
    name = "reinforce_baseline"

    def _advantages(self, returns, states):
        # Un episode d'un seul pas n'a pas de variance a reduire : on le laisse
        # tel quel plutot que de renvoyer un avantage nul, qui annulerait le
        # gradient.
        if returns.numel() < 2:
            return returns
        return returns - returns.mean()
