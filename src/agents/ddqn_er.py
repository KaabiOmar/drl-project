"""Double DQN avec memoire de rejeu uniforme.

Un seul changement par rapport a `DoubleDqnAgent` : `observe()` n'apprend plus
sur la transition qu'il vient de recevoir, il l'empile. Toutes les
`train_every` transitions, on tire un lot uniforme dans toute la memoire.

Deux raisons a cela, toutes deux a savoir expliquer :

**La correlation temporelle.** Deux transitions consecutives se ressemblent
beaucoup. Apprendre dessus dans l'ordre, c'est enchainer des gradients presque
identiques, ce qui fait osciller le reseau au lieu de le faire converger.

**La reutilisation.** Sans memoire, chaque experience sert une fois puis est
perdue. Avec, une transition rare - une victoire, par exemple - continue d'etre
retiree pendant des milliers de pas.

`warmup` evite d'apprendre sur une memoire quasi vide, ou le meme echantillon
reviendrait sans cesse.
"""

from __future__ import annotations

from src.agents.ddqn import DoubleDqnAgent
from src.common.replay_buffer import ReplayBuffer, Transition


class DoubleDqnExperienceReplayAgent(DoubleDqnAgent):
    name = "ddqn_er"

    def __init__(
        self,
        *args,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        warmup: int = 1_000,
        train_every: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.warmup = warmup
        self.train_every = train_every
        self.transitions_seen = 0

    def observe(self, transition: Transition) -> None:
        self.buffer.push(transition)
        self.transitions_seen += 1
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return
        if self.transitions_seen % self.train_every != 0:
            return
        self._learn_on_batch(self.buffer.sample(self.batch_size, self.rng))
