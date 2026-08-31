"""Rejoue un modele sauvegarde et reproduit ses metriques.

Usage :
    python -m src.eval --env bobail --agent dqn --model runs/.../model_100000.pt --games 500

C'est le script que le correcteur doit pouvoir lancer pour confirmer les
chiffres du rapport : le README doit en donner la ligne de commande exacte.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.agents import make_agent
from src.common.metrics import evaluate
from src.common.seeding import set_seed
from src.envs import make_env
from src.train import evaluation_factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation d'un modele entraine")
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--opponent", default="random")
    args = parser.parse_args()

    set_seed(args.seed)
    kwargs = {"seed": args.seed} if args.env in ("tictactoe", "bobail") else {}
    if args.env == "bobail":
        kwargs["opponent"] = args.opponent

    env = make_env(args.env, **kwargs)
    agent = make_agent(args.agent, env, seed=args.seed)
    if args.model is not None:
        agent.load(args.model)

    result = evaluate(agent, evaluation_factory(args.env, kwargs, args.seed), games=args.games)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
