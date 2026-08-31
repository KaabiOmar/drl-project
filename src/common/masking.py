"""Masquage des actions illegales.

C'est le point le plus sensible du projet. Un agent qui doit APPRENDRE a ne pas
jouer illegal gaspille l'essentiel de son budget d'entrainement, en particulier
sur Bobail ou moins de 5 % des 208 actions sont legales a chaque instant.

Regles :
* valeurs (Q-values) -> on force les illegales a -inf AVANT l'argmax et AVANT le
  `max` de la cible temporelle ;
* logits de politique -> on ajoute -1e9 AVANT le softmax, jamais apres.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1e9


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    """Indice de la meilleure action legale."""
    masked = np.where(mask > 0.0, values, -np.inf)
    return int(np.argmax(masked))


def masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    """Valeur de la meilleure action legale, 0.0 si aucune ne l'est."""
    if not (mask > 0.0).any():
        return 0.0
    return float(np.max(np.where(mask > 0.0, values, -np.inf)))


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Distribution de probabilite a support restreint aux actions legales."""
    masked = np.where(mask > 0.0, logits, NEG_INF)
    shifted = masked - masked.max()
    exp = np.exp(shifted) * (mask > 0.0)
    total = exp.sum()
    if total <= 0.0:
        # Repli defensif : ne doit jamais arriver si le masque est non vide.
        return mask / max(mask.sum(), 1.0)
    return exp / total


def apply_mask_torch(logits, mask):
    """Version PyTorch de `masked_softmax`, appliquee aux logits.

    `logits` et `mask` sont des tenseurs de meme forme. On renvoie les logits
    masques, a passer ensuite a `log_softmax` ou `Categorical`.
    """
    import torch

    return torch.where(mask > 0.0, logits, torch.full_like(logits, NEG_INF))
