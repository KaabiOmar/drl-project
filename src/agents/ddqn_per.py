"""Double DQN avec memoire de rejeu prioritaire (Schaul et al., 2016).

Delta par rapport a `DoubleDqnExperienceReplayAgent` :
* remplacer le buffer par `PrioritizedReplayBuffer` (deja fourni) ;
* priorite d'une transition = |erreur TD| + epsilon, elevee a la puissance alpha ;
* ponderer la perte de chaque element du batch par son poids d'importance,
  pour compenser le biais introduit par l'echantillonnage non uniforme ;
* remettre a jour les priorites apres chaque batch via `update_priorities()` ;
* anneler `beta` de 0.4 vers 1.0 au cours de l'entrainement.

Piege classique : oublier la ponderation d'importance. L'agent apprend alors
plus vite au debut puis converge vers une politique biaisee.
"""

from __future__ import annotations

from src.agents.ddqn_er import DoubleDqnExperienceReplayAgent


class DoubleDqnPrioritizedReplayAgent(DoubleDqnExperienceReplayAgent):
    name = "ddqn_per"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DoubleDqnPrioritizedReplayAgent n'est pas encore implemente. "
            "La marche a suivre est decrite dans la docstring du module."
        )
