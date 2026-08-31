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
from dataclasses import asdict
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
    set_seed(args.seed)
    config = json.loads(args.config.read_text()) if args.config else {}

    kwargs = env_kwargs(args.env, args, args.seed)
    env = make_env(args.env, **kwargs)
    agent = make_agent(args.agent, env, seed=args.seed, **config)

    run_dir = args.out / f"{args.env}__{args.agent}__seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), **config}, default=str, indent=2))
    logger = CsvLogger(run_dir / "metrics.csv")

    checkpoints = [c for c in CHECKPOINTS if c <= args.episodes] or [args.episodes]
    if args.episodes not in checkpoints:
        checkpoints.append(args.episodes)

    started = time.perf_counter()
    for episode in range(1, args.episodes + 1):
        if args.agent not in NON_LEARNING_AGENTS:
            run_training_episode(env, agent)
        if episode in checkpoints:
            result = evaluate(agent, lambda: make_env(args.env, **kwargs), games=args.eval_games)
            result.episodes_trained = episode
            logger.log(result)
            agent.save(run_dir / f"model_{episode}.pt")
            elapsed = time.perf_counter() - started
            print(
                f"[{episode:>9}] score={result.mean_score:+.3f} "
                f"win={result.win_rate:.2f} len={result.mean_length:.1f} "
                f"move={result.mean_time_per_move_ms:.2f}ms ({elapsed:.0f}s)"
            )

    logger.close()
    print(f"resultats: {run_dir/'metrics.csv'}")


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
