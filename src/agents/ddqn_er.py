"""Double DQN avec memoire de rejeu uniforme.

Delta par rapport a `DoubleDqnAgent` :
* `observe()` n'apprend plus immediatement : il empile la transition dans un
  `ReplayBuffer` ;
* toutes les `train_every` transitions, tirer un batch uniforme de taille
  `batch_size` et faire UNE descente de gradient dessus ;
* attendre `warmup` transitions avant la premiere mise a jour.

Interet a expliquer : casser la correlation temporelle des transitions et
reutiliser chaque experience plusieurs fois.
"""

from __future__ import annotations

from src.agents.ddqn import DoubleDqnAgent


class DoubleDqnExperienceReplayAgent(DoubleDqnAgent):
    name = "ddqn_er"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DoubleDqnExperienceReplayAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
