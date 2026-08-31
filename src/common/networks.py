"""Reseaux partages par tous les agents neuronaux.

Volontairement minimalistes : un perceptron multicouche suffit pour des etats de
27 a 77 valeurs. Utiliser une architecture identique partout est ce qui rend la
comparaison entre algorithmes honnete.
"""

from __future__ import annotations

from typing import Sequence


def mlp(input_dim: int, output_dim: int, hidden: Sequence[int] = (128, 128)):
    """Perceptron multicouche `input_dim -> hidden... -> output_dim` (ReLU)."""
    import torch.nn as nn

    layers: list = []
    previous = input_dim
    for size in hidden:
        layers += [nn.Linear(previous, size), nn.ReLU()]
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def actor_critic(input_dim: int, output_dim: int, hidden: Sequence[int] = (128, 128)):
    """Deux tetes separees (politique, valeur) partageant la meme architecture.

    Renvoie `(acteur, critique)`. Des reseaux separes plutot qu'un tronc commun :
    plus simple a expliquer en soutenance et plus stable a regler.
    """
    return mlp(input_dim, output_dim, hidden), mlp(input_dim, 1, hidden)
