"""Memoires de rejeu, uniforme et prioritaire.

Une transition stocke aussi le MASQUE de l'etat suivant : sans lui, la cible
temporelle `max_a' Q(s', a')` porte sur des actions illegales et l'apprentissage
diverge silencieusement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.sum_tree import SumTree


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    """Memoire de rejeu uniforme (circulaire)."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.data: list[Transition] = []
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, transition: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.integers(0, len(self.data), size=batch_size)
        return [self.data[i] for i in indices]


class PrioritizedReplayBuffer:
    """Memoire de rejeu prioritaire (Schaul et al., 2016).

    `alpha` regle l'intensite de la priorisation (0 = uniforme), `beta` la
    correction du biais d'importance, typiquement annelee de 0.4 vers 1.0.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-3,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.data: list[Transition | None] = [None] * capacity
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def push(self, transition: Transition) -> None:
        index = self.tree.add(self.max_priority ** self.alpha)
        self.data[index] = transition

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Renvoie (transitions, indices, poids d'importance normalises)."""
        segment = self.tree.total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        batch: list[Transition] = []
        for i in range(batch_size):
            value = rng.uniform(segment * i, segment * (i + 1))
            index, priority = self.tree.sample(value)
            indices[i] = index
            priorities[i] = priority
            batch.append(self.data[index])

        probabilities = priorities / max(self.tree.total, 1e-12)
        weights = (len(self) * probabilities) ** (-self.beta)
        weights /= max(weights.max(), 1e-12)
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for index, error in zip(indices, np.abs(td_errors)):
            priority = float(error) + self.epsilon
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(index), priority ** self.alpha)
