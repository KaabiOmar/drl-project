"""Registre des agents.

Les onze agents demandes par le sujet y figurent tous, y compris ceux qui ne
sont pas encore ecrits : le registre est la liste de reference du projet, et un
agent manquant echoue bruyamment a l'instanciation plutot que silencieusement
dans les resultats.
"""

from __future__ import annotations

from typing import Callable

from src.agents.base import Agent
from src.agents.ddqn import DoubleDqnAgent
from src.agents.ddqn_er import DoubleDqnExperienceReplayAgent
from src.agents.ddqn_per import DoubleDqnPrioritizedReplayAgent
from src.agents.dqn import DqnAgent
from src.agents.mcts_uct import MctsUctAgent
from src.agents.ppo import PpoA2CAgent
from src.agents.random_agent import RandomAgent
from src.agents.random_rollout import RandomRolloutAgent
from src.agents.reinforce import ReinforceAgent
from src.agents.reinforce_baseline import ReinforceMeanBaselineAgent
from src.agents.reinforce_critic import ReinforceCriticBaselineAgent

AGENT_REGISTRY: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "dqn": DqnAgent,
    "ddqn": DoubleDqnAgent,
    "ddqn_er": DoubleDqnExperienceReplayAgent,
    "ddqn_per": DoubleDqnPrioritizedReplayAgent,
    "reinforce": ReinforceAgent,
    "reinforce_baseline": ReinforceMeanBaselineAgent,
    "reinforce_critic": ReinforceCriticBaselineAgent,
    "ppo": PpoA2CAgent,
    "random_rollout": RandomRolloutAgent,
    "mcts_uct": MctsUctAgent,
}

# Agents de planification : ils ont besoin de `env` dans `act()` et n'ont ni
# entrainement ni modele a sauvegarder.
PLANNING_AGENTS = frozenset({"random_rollout", "mcts_uct"})

# Agents sans apprentissage : la boucle d'entrainement doit les court-circuiter.
NON_LEARNING_AGENTS = PLANNING_AGENTS | {"random"}


def make_agent(name: str, env, seed: int | None = None, **config) -> Agent:
    """Instancie un agent en lui injectant les dimensions de l'environnement."""
    if name not in AGENT_REGISTRY:
        raise KeyError(f"agent inconnu: {name!r} (connus: {sorted(AGENT_REGISTRY)})")
    return AGENT_REGISTRY[name](
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        seed=seed,
        **config,
    )


__all__ = ["Agent", "AGENT_REGISTRY", "PLANNING_AGENTS", "NON_LEARNING_AGENTS", "make_agent"]
