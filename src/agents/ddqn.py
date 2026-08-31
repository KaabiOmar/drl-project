"""Double Deep Q-Learning.

Delta par rapport a `DqnAgent` :
* ajouter un RESEAU CIBLE `target_network`, copie periodique du reseau en ligne
  (tous les `target_update` pas), gele entre deux copies ;
* decoupler le choix et l'evaluation de l'action dans la cible :
  `a* = argmax_a' Q_online(s', a')` puis `y = r + gamma * Q_target(s', a*)`.
  C'est ce decouplage qui corrige la surestimation systematique du `max`.
* le masque de `s'` s'applique AU CHOIX de `a*`, sur le reseau en ligne.

A verifier : sans reseau cible, la courbe de score plafonne puis s'effondre. Le
montrer dans le rapport est un tres bon argument.
"""

from __future__ import annotations

from src.agents.dqn import DqnAgent


class DoubleDqnAgent(DqnAgent):
    name = "ddqn"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DoubleDqnAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
