"""PPO, style A2C : acteur-critique a ratio clippe.

Le probleme que PPO resout : dans REINFORCE, rien n'empeche une mise a jour de
deplacer massivement la politique. Un seul lot malchanceux peut detruire ce qui
a ete appris, et comme la methode est on-policy, les donnees suivantes sont
collectees par la politique degradee - l'effondrement s'auto-entretient.

PPO borne le deplacement. On compare la politique courante a celle qui a
collecte les donnees via le ratio `r = pi(a|s) / pi_ancienne(a|s)`, et on
optimise :

    L = min( r * A , clip(r, 1-eps, 1+eps) * A )

Le `min` fait que depasser la borne n'apporte plus rien : le gradient
s'annule des que la politique s'eloigne trop. On peut donc faire PLUSIEURS
passes sur le meme lot sans exploser, ce qui rend PPO bien plus efficace en
donnees que REINFORCE.

Trois details qui comptent :

* **GAE(lambda)** interpole entre l'avantage a un pas (peu de variance, du
  biais) et le retour Monte Carlo (pas de biais, beaucoup de variance) ;
* **le masque est stocke avec la transition** et rejoue tel quel lors des
  passes. Recalculer un masque different rendrait les ratios faux, et le bug
  serait invisible : l'entrainement tournerait, simplement moins bien ;
* **`horizon` doit rester petit ici.** Les parties durent 2 a 10 pas ; avec
  l'horizon de 2048 habituel, 10 000 episodes ne donneraient qu'une poignee de
  mises a jour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.base import Agent
from src.common.masking import apply_mask_torch, masked_argmax
from src.common.networks import actor_critic
from src.common.replay_buffer import Transition


class PpoA2CAgent(Agent):
    name = "ppo"

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        horizon: int = 256,
        epochs: int = 4,
        batch_size: int = 64,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        seed: int | None = None,
        **_: object,
    ) -> None:
        import torch

        self.torch = torch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.rng = np.random.default_rng(seed)

        self.actor, self.critic = actor_critic(state_dim, num_actions, hidden)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate
        )
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        self.states: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.old_log_probs: list[float] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[float] = []

    # --- decision -------------------------------------------------------------

    def act(self, state, mask, env=None, greedy: bool = False) -> int:
        torch = self.torch
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = apply_mask_torch(self.actor(state_tensor), mask_tensor)

        if greedy:
            return masked_argmax(logits.squeeze(0).numpy(), mask)

        distribution = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = distribution.sample()
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(int(action.item()))
        self.old_log_probs.append(float(distribution.log_prob(action).item()))
        return int(action.item())

    # --- apprentissage --------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        self.rewards.append(transition.reward)
        self.next_states.append(transition.next_state)
        self.dones.append(float(transition.done))
        if len(self.rewards) >= self.horizon:
            self._update()

    def end_episode(self) -> None:
        """Rien a faire : PPO se met a jour a l'horizon, pas a la fin d'episode."""

    def _update(self) -> None:
        torch = self.torch
        states = torch.as_tensor(np.array(self.states), dtype=torch.float32)
        masks = torch.as_tensor(np.array(self.masks), dtype=torch.float32)
        actions = torch.as_tensor(self.actions, dtype=torch.int64)
        old_log_probs = torch.as_tensor(self.old_log_probs, dtype=torch.float32)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32)
        next_states = torch.as_tensor(np.array(self.next_states), dtype=torch.float32)
        dones = torch.as_tensor(self.dones, dtype=torch.float32)

        with torch.no_grad():
            values = self.critic(states).squeeze(1)
            next_values = self.critic(next_states).squeeze(1)

        advantages = self._generalized_advantages(rewards, values, next_values, dones)
        returns = advantages + values
        # Normaliser les avantages : ils changent d'echelle d'un lot a l'autre,
        # ce qui rendrait le taux d'apprentissage impossible a regler.
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        size = len(self.rewards)
        for _ in range(self.epochs):
            order = self.rng.permutation(size)
            for start in range(0, size, self.batch_size):
                index = torch.as_tensor(order[start:start + self.batch_size], dtype=torch.int64)

                logits = apply_mask_torch(self.actor(states[index]), masks[index])
                distribution = torch.distributions.Categorical(logits=logits)
                log_probs = distribution.log_prob(actions[index])

                ratio = torch.exp(log_probs - old_log_probs[index])
                unclipped = ratio * normalized[index]
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * normalized[index]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = torch.nn.functional.mse_loss(
                    self.critic(states[index]).squeeze(1), returns[index]
                )
                entropy = distribution.entropy().mean()

                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()

        self._reset_rollout()

    def _generalized_advantages(self, rewards, values, next_values, dones):
        """GAE(lambda), calcule a rebours sur le rollout."""
        torch = self.torch
        advantages = torch.zeros_like(rewards)
        running = 0.0
        for step in reversed(range(len(rewards))):
            not_done = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_values[step] * not_done - values[step]
            running = delta + self.gamma * self.gae_lambda * not_done * running
            advantages[step] = running
        return advantages

    # --- persistance ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
