"""Journalisation et protocole d'evaluation.

Le syllabus demande les metriques de la POLITIQUE OBTENUE, pas de la politique
d'entrainement. `evaluate()` est donc le seul chemin autorise pour produire un
chiffre de rapport : il gele l'agent en mode glouton (`greedy=True`), ce qui
retire l'exploration (epsilon = 0, argmax au lieu d'echantillonnage).
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(slots=True)
class EvalResult:
    episodes_trained: int
    mean_score: float
    std_score: float
    mean_length: float
    mean_time_per_move_ms: float
    win_rate: float
    loss_rate: float
    draw_rate: float


def evaluate(agent, env_factory: Callable[[int], object], games: int = 100) -> EvalResult:
    """Joue `games` parties avec la politique gelee et agrege les metriques.

    `env_factory` recoit le NUMERO de la partie et doit renvoyer un
    environnement dont la graine en depend. Une fabrique qui ignore ce numero
    rejouerait `games` fois la meme partie : la moyenne serait celle d'un seul
    tirage et l'ecart-type nul, ce qui donne des metriques d'apparence parfaite
    et sans aucune valeur.
    """
    scores, lengths, move_times = [], [], []
    for game in range(games):
        env = env_factory(game)
        env.reset()
        steps = 0
        while not env.is_game_over():
            state = env.state_description()
            mask = env.available_actions_mask()
            if mask.sum() == 0:
                break
            started = time.perf_counter()
            action = agent.act(state, mask, env=env, greedy=True)
            move_times.append((time.perf_counter() - started) * 1000.0)
            env.step(action)
            steps += 1
        scores.append(env.score())
        lengths.append(steps)

    scores_array = np.asarray(scores, dtype=np.float64)
    return EvalResult(
        episodes_trained=0,
        mean_score=float(scores_array.mean()),
        std_score=float(scores_array.std()),
        mean_length=float(np.mean(lengths)),
        mean_time_per_move_ms=float(np.mean(move_times)) if move_times else 0.0,
        win_rate=float((scores_array > 0).mean()),
        loss_rate=float((scores_array < 0).mean()),
        draw_rate=float((scores_array == 0).mean()),
    )


class CsvLogger:
    """Journal CSV a en-tete auto-decouvert, une ligne par point de mesure."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: csv.DictWriter | None = None
        self._file = None

    def log(self, record: EvalResult | dict) -> None:
        row = asdict(record) if isinstance(record, EvalResult) else dict(record)
        if self._writer is None:
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
