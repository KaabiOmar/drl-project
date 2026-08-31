"""REINFORCE : gradient de politique Monte Carlo.

Rupture avec la famille DQN : on n'apprend plus une valeur d'action pour en
deduire une politique, on parametre DIRECTEMENT la politique et on remonte le
gradient qui augmente la probabilite des actions ayant mene a un bon retour.

    grad J = E[ sum_t grad log pi(a_t | s_t) * G_t ]

Trois consequences a savoir defendre :

* la mise a jour n'a lieu qu'a la FIN de l'episode, puisque `G_t` demande de
  connaitre toutes les recompenses qui suivent. D'ou une variance elevee : deux
  episodes identiques au debut mais differents a la fin produisent des gradients
  opposes sur les memes premiers pas ;
* la politique est stochastique par construction, l'exploration vient de
  l'echantillonnage et non d'un epsilon ajoute a la main ;
* le masque s'applique aux LOGITS avant le softmax. Ajouter -1e9 revient a
  donner une probabilite nulle aux actions illegales sans casser le gradient.

En evaluation (`greedy=True`) on prend l'action la plus probable : c'est la
politique obtenue, celle dont le sujet demande les metriques.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.base import Agent
from src.common.masking import apply_mask_torch, masked_argmax
from src.common.networks import mlp
from src.common.replay_buffer import Transition


class ReinforceAgent(Agent):
    name = "reinforce"

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        seed: int | None = None,
        **_: object,
    ) -> None:
        import torch

        self.torch = torch
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.policy = mlp(state_dim, num_actions, hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.log_probs: list = []
        self.states: list[np.ndarray] = []
        self.rewards: list[float] = []

    # --- decision -------------------------------------------------------------

    def act(self, state, mask, env=None, greedy: bool = False) -> int:
        torch = self.torch
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

        if greedy:
            with torch.no_grad():
                logits = apply_mask_torch(self.policy(state_tensor), mask_tensor)
            return masked_argmax(logits.squeeze(0).numpy(), mask)

        logits = apply_mask_torch(self.policy(state_tensor), mask_tensor)
        distribution = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        return int(action.item())

    # --- apprentissage --------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        self.states.append(transition.state)
        self.rewards.append(transition.reward)

    def end_episode(self) -> None:
        if not self.log_probs:
            return
        torch = self.torch

        returns = torch.as_tensor(self._discounted_returns(), dtype=torch.float32)
        states = torch.as_tensor(np.array(self.states), dtype=torch.float32)
        log_probs = torch.stack(self.log_probs)

        advantages = self._advantages(returns, states)
        policy_loss = -(log_probs * advantages).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self._update_auxiliary(states, returns)
        self._reset_episode()

    def _discounted_returns(self) -> list[float]:
        """G_t = r_t + gamma * G_{t+1}, calcule a rebours."""
        returns, running = [], 0.0
        for reward in reversed(self.rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns.reverse()
        return returns

    def _advantages(self, returns, states):
        """REINFORCE nu : le retour brut sert de signal."""
        return returns

    def _update_auxiliary(self, states, returns) -> None:
        """Point d'extension pour les variantes qui entrainent un critique."""

    # --- persistance ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy"])
