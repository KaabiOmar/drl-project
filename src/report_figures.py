"""Figures et tableau de synthese du rapport, a partir des runs.

Lit tous les `runs/*/metrics.csv`, agrege les graines et produit :
* une courbe par environnement : score de la politique obtenue en fonction du
  nombre d'episodes d'entrainement, une ligne par agent, la zone ombree
  couvrant l'ecart entre graines ;
* un histogramme des scores finaux par environnement ;
* `reports/summary.csv`, le tableau de tous les chiffres demandes par le sujet.

Usage :
    python -m src.report_figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_runs(runs_dir: Path) -> dict[tuple[str, str], dict[int, list[dict]]]:
    """Indexe les mesures par (environnement, agent) puis par nombre d'episodes."""
    data: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for metrics_path in sorted(runs_dir.glob("*/metrics.csv")):
        parts = metrics_path.parent.name.split("__")
        if len(parts) != 3:
            continue
        env_name, agent_name, _ = parts
        with metrics_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                episodes = int(row["episodes_trained"])
                data[(env_name, agent_name)][episodes].append(
                    {key: float(value) for key, value in row.items()}
                )
    return data


def write_summary(data, output: Path) -> None:
    """Tableau de synthese : une ligne par (environnement, agent, point de mesure)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["environnement", "agent", "episodes", "graines", "score_moyen",
             "ecart_entre_graines", "longueur_moyenne", "ms_par_coup", "taux_victoire"]
        )
        for (env_name, agent_name), by_episode in sorted(data.items()):
            for episodes, rows in sorted(by_episode.items()):
                scores = [row["mean_score"] for row in rows]
                writer.writerow([
                    env_name,
                    agent_name,
                    episodes,
                    len(rows),
                    f"{sum(scores)/len(scores):.3f}",
                    f"{max(scores)-min(scores):.3f}",
                    f"{sum(r['mean_length'] for r in rows)/len(rows):.1f}",
                    f"{sum(r['mean_time_per_move_ms'] for r in rows)/len(rows):.3f}",
                    f"{sum(r['win_rate'] for r in rows)/len(rows):.3f}",
                ])


def plot_learning_curves(data, figures_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    envs = sorted({env_name for env_name, _ in data})
    for env_name in envs:
        figure, axes = plt.subplots(figsize=(8, 5))
        for (name, agent_name), by_episode in sorted(data.items()):
            if name != env_name:
                continue
            episodes = sorted(by_episode)
            means = [
                sum(r["mean_score"] for r in by_episode[e]) / len(by_episode[e]) for e in episodes
            ]
            axes.plot(episodes, means, marker="o", label=agent_name)
            lows = [min(r["mean_score"] for r in by_episode[e]) for e in episodes]
            highs = [max(r["mean_score"] for r in by_episode[e]) for e in episodes]
            axes.fill_between(episodes, lows, highs, alpha=0.15)

        axes.set_xscale("log")
        axes.set_xlabel("episodes d'entrainement")
        axes.set_ylabel("score moyen de la politique obtenue")
        axes.set_title(f"{env_name} - apprentissage")
        axes.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        axes.grid(alpha=0.25)
        axes.legend(fontsize=8)
        path = figures_dir / f"apprentissage_{env_name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures et synthese du rapport")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    data = read_runs(args.runs)
    if not data:
        print(f"aucun run trouve dans {args.runs}/")
        return

    summary = args.out / "summary.csv"
    write_summary(data, summary)
    print(f"tableau : {summary}")
    for path in plot_learning_curves(data, args.out / "figures"):
        print(f"figure  : {path}")


if __name__ == "__main__":
    main()
