"""Campagne d'experimentation : la matrice (environnement x agent x graine).

Le sujet demande de comparer 11 agents sur 4 environnements. Fait a la main,
c'est 44 lignes de commande a ne pas se tromper, et des resultats impossibles a
rattacher a leur configuration trois jours plus tard. Ce script lance la matrice
complete, ecrit un run par combinaison sous `runs/`, et continue si une
combinaison echoue - un algorithme non implemente ne doit pas arreter la
campagne.

Usage :
    python -m src.experiment --envs line_world,tictactoe --agents random,dqn \\
        --seeds 0,1,2 --episodes 10000

    python -m src.experiment --dry-run          # affiche la matrice sans rien lancer

Trois graines par configuration : sans repetition, un ecart entre deux
algorithmes n'est pas interpretable, c'est peut-etre juste du bruit.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from src.agents import AGENT_REGISTRY
from src.envs import ENV_REGISTRY
from src.train import train_one

DEFAULT_SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Campagne d'entrainement multi-agents")
    parser.add_argument("--envs", default=",".join(ENV_REGISTRY))
    parser.add_argument("--agents", default=",".join(AGENT_REGISTRY))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--opponent", default="random", help="bobail : random | heuristic")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(directory: Path, agent: str) -> dict:
    """Hyperparametres de l'agent, depuis `configs/<agent>.json` s'il existe."""
    path = directory / f"{agent}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    args = parse_args()
    envs = [name.strip() for name in args.envs.split(",") if name.strip()]
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    combinations = [(e, a, s) for e in envs for a in agents for s in seeds]
    print(f"{len(combinations)} runs : {len(envs)} environnements x {len(agents)} agents x {len(seeds)} graines")
    print(f"{args.episodes} episodes chacun\n")

    if args.dry_run:
        for env_name, agent_name, seed in combinations:
            print(f"  {env_name:<12} {agent_name:<18} graine {seed}")
        return

    done, failed = [], []
    started = time.perf_counter()
    for index, (env_name, agent_name, seed) in enumerate(combinations, start=1):
        label = f"{env_name}/{agent_name}/graine {seed}"
        print(f"[{index}/{len(combinations)}] {label}")
        try:
            run_dir = train_one(
                env_name=env_name,
                agent_name=agent_name,
                episodes=args.episodes,
                seed=seed,
                eval_games=args.eval_games,
                opponent=args.opponent,
                config=load_config(args.configs, agent_name),
                out=args.out,
                verbose=False,
            )
            done.append(run_dir)
        except NotImplementedError:
            # Agent pas encore ecrit : on le note et on continue la campagne.
            print(f"    ignore : {agent_name} n'est pas encore implemente")
            failed.append((label, "non implemente"))
        except Exception as error:  # noqa: BLE001 - on veut la campagne complete
            print(f"    ECHEC : {type(error).__name__}: {error}")
            traceback.print_exc()
            failed.append((label, f"{type(error).__name__}: {error}"))

    elapsed = time.perf_counter() - started
    print(f"\n{len(done)} runs termines, {len(failed)} en echec, en {elapsed/60:.1f} min")
    for label, reason in failed:
        print(f"  - {label} : {reason}")


if __name__ == "__main__":
    main()
