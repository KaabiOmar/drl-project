"""Arbre de sommes, structure de base du Prioritized Experience Replay.

Permet un echantillonnage proportionnel a la priorite en O(log n) au lieu de
O(n). Chaque feuille porte la priorite d'une transition, chaque noeud interne la
somme de ses deux enfants.
"""

from __future__ import annotations

import numpy as np


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.size = 0
        self.cursor = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """Insere une priorite et renvoie l'indice de donnee associe."""
        index = self.cursor
        self.update(index, priority)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def update(self, index: int, priority: float) -> None:
        node = index + self.capacity - 1
        delta = priority - self.tree[node]
        self.tree[node] = priority
        while node > 0:
            node = (node - 1) // 2
            self.tree[node] += delta

    def sample(self, value: float) -> tuple[int, float]:
        """Renvoie (indice de donnee, priorite) pour un tirage `value` dans [0, total]."""
        node = 0
        while node < self.capacity - 1:
            left = 2 * node + 1
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        index = node - self.capacity + 1
        return index, float(self.tree[node])
