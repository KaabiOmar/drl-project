"""PPO, style A2C (acteur-critique avec ratio clippe).

A implementer :
* collecter un rollout de `horizon` pas (etats, actions, log_probs, valeurs,
  recompenses, masques) ;
* calculer les avantages par GAE(lambda) ;
* faire `epochs` passes de mini-batchs sur ce rollout, avec la perte clippee
  `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`, plus la perte de valeur et un
  bonus d'entropie ;
* jeter le rollout et recommencer (on-policy).

Le masque doit etre stocke avec les transitions : le recalcul des log_probs lors
des epochs doit utiliser EXACTEMENT le meme masque qu'a la collecte, sinon les
ratios sont faux.
"""

from __future__ import annotations

from src.agents.base import Agent


class PpoA2CAgent(Agent):
    name = "ppo"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "PpoA2CAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
