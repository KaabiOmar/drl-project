"""Double Deep Q-Learning.

Deux changements par rapport a `DqnAgent`, et il faut savoir expliquer pourquoi
chacun existe.

**Le reseau cible.** Dans le DQN simple, la cible `r + gamma * max Q(s')` est
calculee avec le reseau qu'on est justement en train de modifier : on poursuit
une cible mouvante. On garde donc une copie gelee du reseau, recopiee tous les
`target_update` apprentissages, pour stabiliser cette cible.

**Le decouplage choix / evaluation.** Le `max` du DQN simple choisit l'action ET
evalue sa valeur avec le meme reseau. Comme les erreurs d'estimation sont
aleatoires, le `max` tombe systematiquement sur les actions SURESTIMEES, et ce
biais s'accumule. Double DQN choisit l'action avec le reseau en ligne, puis
l'evalue avec le reseau cible :

    a* = argmax_a' Q_online(s', a')
    y  = r + gamma * Q_cible(s', a*)

`_learn_on_batch` est ecrit pour un lot de taille quelconque : les deux agents
qui heritent de celui-ci (rejeu uniforme et prioritaire) n'ont plus qu'a fournir
le lot et, pour le prioritaire, les poids d'importance.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src.agents.dqn import DqnAgent
from src.common.replay_buffer import Transition


class DoubleDqnAgent(DqnAgent):
    name = "ddqn"

    def __init__(self, *args, target_update: int = 500, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_network = copy.deepcopy(self.network)
        self.target_network.eval()
        self.target_update = target_update
        self.updates = 0

    def observe(self, transition: Transition) -> None:
        """Apprentissage en ligne : un lot d'une seule transition."""
        self._learn_on_batch([transition])

    def _learn_on_batch(self, batch: list[Transition], weights=None):
        """Une descente de gradient sur un lot. Renvoie les erreurs TD."""
        torch = self.torch
        states = torch.as_tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        next_masks = torch.as_tensor(np.array([t.next_mask for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor([float(t.done) for t in batch], dtype=torch.float32)

        predicted = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            online_next = self.network(next_states)
            # Le choix de a* se fait sur les seules actions legales de s'.
            online_next = torch.where(next_masks > 0.0, online_next, torch.full_like(online_next, -1e9))
            best_actions = online_next.argmax(dim=1)
            target_next = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # Terme futur annule sur les etats terminaux et sur les etats sans
            # action legale : sans cela on bootstrappe sur une valeur inventee.
            bootstrap = (1.0 - dones) * (next_masks.sum(dim=1) > 0).float()
            targets = rewards + self.gamma * target_next * bootstrap

        errors = targets - predicted
        losses = torch.nn.functional.smooth_l1_loss(predicted, targets, reduction="none")
        if weights is not None:
            losses = losses * torch.as_tensor(weights, dtype=torch.float32)

        self.optimizer.zero_grad()
        losses.mean().backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return errors.detach().numpy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "network": self.network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint.get("target_network", checkpoint["network"]))
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
