"""Deep Q-Learning en ligne (sans memoire de rejeu).

Version de reference de la famille "value based" : c'est le fichier a lire et a
comprendre avant d'ecrire Double DQN, DDQN + ER et DDQN + PER, qui n'en sont que
des variations.

Points a savoir expliquer en soutenance :
* pourquoi la cible `r + gamma * max_a' Q(s', a')` est BOOTSTRAPPEE (elle depend
  du reseau qu'on entraine, d'ou l'instabilite que Double DQN corrige) ;
* pourquoi le `max` doit etre pris sur les seules actions LEGALES de s' ;
* pourquoi on annule le terme futur quand l'episode est termine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.base import Agent
from src.common.masking import masked_argmax
from src.common.networks import mlp
from src.common.replay_buffer import Transition


class DqnAgent(Agent):
    name = "dqn"

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
        **_: object,
    ) -> None:
        import torch

        self.torch = torch
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.network = mlp(state_dim, num_actions, hidden)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    # --- decision -------------------------------------------------------------

    def act(self, state, mask, env=None, greedy: bool = False) -> int:
        legal = np.flatnonzero(mask > 0.0)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        return masked_argmax(self._q_values(state), mask)

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(state, dtype=self.torch.float32).unsqueeze(0)
            return self.network(tensor).squeeze(0).numpy()

    # --- apprentissage --------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        torch = self.torch
        state = torch.as_tensor(transition.state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.as_tensor(transition.next_state, dtype=torch.float32).unsqueeze(0)
        next_mask = torch.as_tensor(transition.next_mask, dtype=torch.float32).unsqueeze(0)

        predicted = self.network(state)[0, transition.action]

        with torch.no_grad():
            if transition.done or next_mask.sum() == 0:
                target = torch.as_tensor(transition.reward, dtype=torch.float32)
            else:
                next_q = self.network(next_state)
                next_q = torch.where(next_mask > 0.0, next_q, torch.full_like(next_q, -float("inf")))
                target = transition.reward + self.gamma * next_q.max()

        loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # --- persistance ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"network": self.network.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path: str | Path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.network.load_state_dict(checkpoint["network"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
