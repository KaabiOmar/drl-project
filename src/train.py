"""Boucle d'entrainement unique, partagee par tous les couples (env, agent).

Usage :
    python -m src.train --env bobail --agent dqn --episodes 100000 --seed 0

Les points de mesure suivent le sujet : 1 000, 10 000, 100 000 et 1 000 000
episodes. A chaque point on GELE la politique et on joue `--eval-games` parties
en mode glouton : ce sont ces chiffres, et eux seuls, qui vont dans le rapport.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.agents import NON_LEARNING_AGENTS, make_agent
from src.common.metrics import CsvLogger, evaluate
from src.common.replay_buffer import Transition
from src.common.seeding import set_seed
from src.envs import make_env

CHECKPOINTS = (1_000, 10_000, 100_000, 1_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement d'un agent sur un environnement")
    parser.add_argument("--env", required=True, help="line_world | grid_world | tictactoe | bobail")
    parser.add_argument("--agent", required=True, help="nom dans AGENT_REGISTRY")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", help="bobail uniquement: random | heuristic")
    parser.add_argument("--config", type=Path, default=None, help="JSON d'hyperparametres")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    return parser.parse_args()


def env_kwargs(name: str, args: argparse.Namespace, seed: int) -> dict:
    kwargs: dict = {}
    if name in ("tictactoe", "bobail"):
        kwargs["seed"] = seed
    if name == "bobail":
        kwargs["opponent"] = args.opponent
    return kwargs


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    run_dir = train_one(
        env_name=args.env,
        agent_name=args.agent,
        episodes=args.episodes,
        seed=args.seed,
        eval_games=args.eval_games,
        opponent=args.opponent,
        config=config,
        out=args.out,
    )
    print(f"resultats: {run_dir/'metrics.csv'}")


def train_one(
    env_name: str,
    agent_name: str,
    episodes: int,
    seed: int = 0,
    eval_games: int = 100,
    opponent: str = "random",
    config: dict | None = None,
    out: Path = Path("runs"),
    verbose: bool = True,
) -> Path:
    """Entraine un couple (environnement, agent) et ecrit ses metriques.

    Point d'entree partage par la ligne de commande et par `src.experiment`,
    pour qu'une campagne complete et un lancement manuel produisent exactement
    les memes fichiers.
    """
    config = config or {}
    set_seed(seed)

    kwargs: dict = {}
    if env_name in ("tictactoe", "bobail"):
        kwargs["seed"] = seed
    if env_name == "bobail":
        kwargs["opponent"] = opponent

    env = make_env(env_name, **kwargs)
    agent = make_agent(agent_name, env, seed=seed, **config)

    run_dir = out / f"{env_name}__{agent_name}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "env": env_name,
                "agent": agent_name,
                "episodes": episodes,
                "seed": seed,
                "opponent": opponent,
                "eval_games": eval_games,
                **config,
            },
            default=str,
            indent=2,
        )
    )
    logger = CsvLogger(run_dir / "metrics.csv")

    checkpoints = [c for c in CHECKPOINTS if c <= episodes] or [episodes]
    if episodes not in checkpoints:
        checkpoints.append(episodes)

    started = time.perf_counter()
    for episode in range(1, episodes + 1):
        if agent_name not in NON_LEARNING_AGENTS:
            run_training_episode(env, agent)
        if episode in checkpoints:
            result = evaluate(agent, lambda: make_env(env_name, **kwargs), games=eval_games)
            result.episodes_trained = episode
            logger.log(result)
            agent.save(run_dir / f"model_{episode}.pt")
            if verbose:
                elapsed = time.perf_counter() - started
                print(
                    f"[{episode:>9}] score={result.mean_score:+.3f} "
                    f"win={result.win_rate:.2f} len={result.mean_length:.1f} "
                    f"move={result.mean_time_per_move_ms:.2f}ms ({elapsed:.0f}s)"
                )

    logger.close()
    return run_dir


def run_training_episode(env, agent) -> None:
    """Un episode d'entrainement : l'exploration est ACTIVE ici, jamais en evaluation."""
    env.reset()
    state = env.state_description()
    mask = env.available_actions_mask()
    while not env.is_game_over():
        if mask.sum() == 0:
            break
        previous_score = env.score()
        action = agent.act(state, mask, env=env, greedy=False)
        env.step(action)

        next_state = env.state_description()
        next_mask = env.available_actions_mask()
        agent.observe(
            Transition(
                state=state,
                action=int(action),
                reward=float(env.score() - previous_score),
                next_state=next_state,
                next_mask=next_mask,
                done=env.is_game_over(),
            )
        )
        state, mask = next_state, next_mask
    agent.end_episode()


if __name__ == "__main__":
    main()
