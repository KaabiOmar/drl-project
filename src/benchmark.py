"""Debit de simulation : parties par seconde en jeu totalement aleatoire.

Livrable explicite de l'etape intermediaire. C'est aussi le chiffre qui permet
de dire, preuve a l'appui, combien d'episodes sont atteignables dans le temps
imparti - et donc de justifier dans le rapport pourquoi 1 000 000 de parties
n'est pas atteint sur Bobail.

Usage :
    python -m src.benchmark --seconds 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.envs import ENV_REGISTRY, make_env


def measure(name: str, seconds: float, rng: np.random.Generator) -> tuple[int, float, float]:
    """Renvoie (parties jouees, parties/seconde, longueur moyenne)."""
    kwargs = {"seed": int(rng.integers(1 << 30))} if name in ("tictactoe", "bobail") else {}
    env = make_env(name, **kwargs)
    games, steps = 0, 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        env.reset()
        while not env.is_game_over():
            legal = np.flatnonzero(env.available_actions_mask() > 0.0)
            if legal.size == 0:
                break
            env.step(int(rng.choice(legal)))
            steps += 1
        games += 1
    elapsed = time.perf_counter() - started
    return games, games / elapsed, steps / max(games, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debit de simulation par environnement")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"{'environnement':<14}{'parties':>10}{'parties/s':>14}{'pas/partie':>13}")
    for name in ENV_REGISTRY:
        games, rate, length = measure(name, args.seconds, rng)
        print(f"{name:<14}{games:>10}{rate:>14.1f}{length:>13.1f}")


if __name__ == "__main__":
    main()
