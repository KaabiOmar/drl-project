"""REINFORCE avec baseline apprise par un critique.

La baseline constante du variant precedent ignore l'etat : elle retire la meme
valeur a un pas prometteur et a un pas desespere. Ici la baseline devient un
reseau `V(s)` entraine en parallele a predire le retour observe. L'avantage
devient :

    A_t = G_t - V(s_t)

soit "ce pas a-t-il fait mieux que ce qu'on attendait de cet etat ?". C'est le
premier pas vers l'acteur-critique, et donc vers PPO.

Deux precautions dans le code, toutes deux classiques a l'oral :

* `.detach()` sur l'avantage. Sans lui, le gradient de la politique remonterait
  dans le critique et l'entrainerait a minimiser l'avantage plutot qu'a predire
  le retour ;
* deux optimiseurs distincts. Les deux pertes n'ont pas la meme echelle, et il
  faut pouvoir les suivre separement dans les logs.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.reinforce import ReinforceAgent
from src.common.networks import mlp


class ReinforceCriticBaselineAgent(ReinforceAgent):
    name = "reinforce_critic"

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        critic_learning_rate: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(state_dim, num_actions, hidden=hidden, **kwargs)
        self.critic = mlp(state_dim, 1, hidden)
        self.critic_optimizer = self.torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.last_critic_loss = 0.0

    def _advantages(self, returns, states):
        with self.torch.no_grad():
            values = self.critic(states).squeeze(1)
        return returns - values

    def _update_auxiliary(self, states, returns) -> None:
        """Regression du critique sur les retours observes."""
        predicted = self.critic(states).squeeze(1)
        loss = self.torch.nn.functional.mse_loss(predicted, returns)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.last_critic_loss = float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}, path
        )

    def load(self, path: str | Path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
