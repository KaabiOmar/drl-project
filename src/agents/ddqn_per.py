"""Double DQN avec memoire de rejeu prioritaire (Schaul et al., 2016).

Idee : toutes les transitions ne s'apprennent pas aussi vite. Celles dont
l'erreur TD est grande sont celles sur lesquelles le reseau se trompe encore, et
donc celles dont il a le plus a apprendre. On les tire plus souvent.

Trois mecanismes, tous necessaires :

**La priorite.** `p = |erreur TD| + epsilon`, elevee a la puissance `alpha`.
Le `epsilon` garantit qu'aucune transition ne tombe a probabilite nulle,
`alpha = 0` redonne exactement le tirage uniforme.

**La ponderation d'importance.** Tirer non uniformement biaise l'esperance du
gradient. On compense en ponderant chaque element par
`w = (N * P(i))^-beta`, normalise par le maximum du lot. C'est le piege
classique : sans cette ponderation, l'agent apprend plus vite au debut puis
converge vers une politique biaisee.

**L'annelage de beta.** On demarre a 0.4 et on monte vers 1.0 : au debut la
vitesse compte plus que le biais, a la fin l'inverse.

Apres chaque lot, les priorites sont remises a jour avec les nouvelles erreurs.
"""

from __future__ import annotations

from src.agents.ddqn_er import DoubleDqnExperienceReplayAgent
from src.common.replay_buffer import PrioritizedReplayBuffer, Transition


class DoubleDqnPrioritizedReplayAgent(DoubleDqnExperienceReplayAgent):
    name = "ddqn_per"

    def __init__(
        self,
        *args,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.buffer = PrioritizedReplayBuffer(self.buffer.capacity, alpha=alpha, beta=beta)
        self.beta_start = beta
        self.beta_end = beta_end
        self.beta_steps = beta_steps

    def observe(self, transition: Transition) -> None:
        self.buffer.push(transition)
        self.transitions_seen += 1
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return
        if self.transitions_seen % self.train_every != 0:
            return

        progress = min(1.0, self.transitions_seen / self.beta_steps)
        self.buffer.beta = self.beta_start + progress * (self.beta_end - self.beta_start)

        batch, indices, weights = self.buffer.sample(self.batch_size, self.rng)
        errors = self._learn_on_batch(batch, weights=weights)
        self.buffer.update_priorities(indices, errors)
