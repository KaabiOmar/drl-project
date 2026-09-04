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

    # Ces figures sont reduites a la largeur d'une page A4 dans le rapport : avec
    # la typo par defaut, la legende devient illisible. On la grossit ici.
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    envs = sorted({env_name for env_name, _ in data})
    for env_name in envs:
        # Deux panneaux : le score appris a gauche, la STABILITE entre graines a
        # droite. Le second est souvent plus parlant que le premier : deux
        # algorithmes au meme score final ne se valent pas si l'un depend de la
        # graine et l'autre non.
        figure, (gauche, droite) = plt.subplots(1, 2, figsize=(13, 5))

        # Sur un environnement sature, plusieurs agents ont EXACTEMENT la meme
        # courbe et le dernier trace recouvre les autres : sur Line World, huit
        # agents sont a +1,00 partout, dont Random Rollout, invisible sous
        # reinforce_critic. On alterne les styles de trait pour que les courbes
        # superposees se devinent, et on annote leur nombre.
        styles = ["-", "--", ":", "-."]
        signatures = {}
        courbes = []
        for (name, agent_name), by_episode in sorted(data.items()):
            if name != env_name:
                continue
            episodes = sorted(by_episode)
            means, lows, highs, spreads = [], [], [], []
            for e in episodes:
                scores = [r["mean_score"] for r in by_episode[e]]
                means.append(sum(scores) / len(scores))
                lows.append(min(scores))
                highs.append(max(scores))
                spreads.append(max(scores) - min(scores))
            courbes.append((agent_name, episodes, means, lows, highs, spreads))
            signatures.setdefault(tuple(round(m, 3) for m in means), []).append(agent_name)

        for index, (agent_name, episodes, means, lows, highs, spreads) in enumerate(courbes):
            style = styles[index % len(styles)]
            trace, = gauche.plot(episodes, means, marker="o", linestyle=style,
                                 label=agent_name)
            gauche.fill_between(episodes, lows, highs, alpha=0.13, color=trace.get_color())
            droite.plot(episodes, spreads, marker="o", linestyle=style,
                        color=trace.get_color(), label=agent_name)

        # Annotation des groupes de courbes confondues, du plus gros au plus petit.
        groupes = sorted(((len(a), sig) for sig, a in signatures.items() if len(a) > 1),
                         reverse=True)
        for rang, (combien, sig) in enumerate(groupes[:2]):
            episodes = courbes[0][1]
            gauche.annotate(f"{combien} agents superposes",
                            xy=(episodes[-1], sig[-1]),
                            xytext=(-12, -26 - 22 * rang), textcoords="offset points",
                            ha="right", fontsize=11, style="italic", color="#444444",
                            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.9))

        for axe, titre, ylabel in (
            (gauche, "apprentissage", "score moyen de la politique obtenue"),
            (droite, "stabilite entre graines", "ecart entre la meilleure et la pire graine"),
        ):
            axe.set_xscale("log")
            axe.set_xlabel("episodes d'entrainement")
            axe.set_ylabel(ylabel)
            axe.set_title(f"{env_name} - {titre}")
            axe.grid(alpha=0.25)
        gauche.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        droite.set_ylim(bottom=0.0)

        # Legende commune, hors des axes : sur 11 agents elle masquait les courbes.
        poignees, etiquettes = gauche.get_legend_handles_labels()
        figure.legend(poignees, etiquettes, loc="lower center", ncol=4, fontsize=13,
                      frameon=False, bbox_to_anchor=(0.5, -0.03))
        path = figures_dir / f"apprentissage_{env_name}.png"
        figure.tight_layout(rect=(0, 0.16, 1, 1))
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
