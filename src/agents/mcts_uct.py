"""Monte Carlo Tree Search, variante UCT.

Implementation "open loop" : l'environnement etant stochastique de notre point
de vue (l'adversaire joue a l'interieur de `step()`), on ne memorise pas l'etat
dans les noeuds. A chaque iteration on repart d'un clone de la racine et on
rejoue le chemin d'actions. Les statistiques d'un noeud moyennent donc sur les
reponses possibles de l'adversaire, ce qui est exactement ce que l'on veut ici.

Les quatre phases classiques : selection (UCB1), expansion, simulation
(rollout aleatoire), retropropagation.
"""

from __future__ import annotations

import math

import numpy as np

from src.agents.base import Agent


class _Node:
    __slots__ = ("children", "visits", "value_sum")

    def __init__(self) -> None:
        self.children: dict[int, "_Node"] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MctsUctAgent(Agent):
    name = "mcts_uct"

    def __init__(
        self,
        iterations: int = 200,
        exploration: float = 1.41,
        seed: int | None = None,
        **_: object,
    ) -> None:
        self.iterations = iterations
        self.exploration = exploration
        self.rng = np.random.default_rng(seed)

    def act(self, state, mask, env=None, greedy: bool = False) -> int:
        if env is None:
            raise ValueError("MctsUctAgent a besoin de `env` pour simuler")
        root = _Node()
        for _ in range(self.iterations):
            self._iterate(env, root)

        legal = np.flatnonzero(mask > 0.0)
        visited = [(action, root.children[action].visits) for action in legal if action in root.children]
        if not visited:
            return int(self.rng.choice(legal))
        # Critere robuste : on joue l'action la plus VISITEE, pas la mieux notee.
        return int(max(visited, key=lambda item: item[1])[0])

    def _iterate(self, env, root: _Node) -> None:
        simulation = env.clone()
        node = root
        path = [root]

        while not simulation.is_game_over():
            legal = np.flatnonzero(simulation.available_actions_mask() > 0.0)
            if legal.size == 0:
                break
            unexplored = [int(a) for a in legal if a not in node.children]
            if unexplored:
                action = int(self.rng.choice(unexplored))
                child = _Node()
                node.children[action] = child
                simulation.step(action)
                node = child
                path.append(child)
                break
            action = self._select(node, legal)
            simulation.step(action)
            node = node.children[action]
            path.append(node)

        value = self._rollout(simulation)
        for visited in path:
            visited.visits += 1
            visited.value_sum += value

    def _select(self, node: _Node, legal: np.ndarray) -> int:
        log_parent = math.log(max(node.visits, 1))
        best_action, best_score = int(legal[0]), -math.inf
        for action in legal:
            child = node.children[int(action)]
            exploit = child.mean_value
            explore = self.exploration * math.sqrt(log_parent / max(child.visits, 1))
            score = exploit + explore
            if score > best_score:
                best_action, best_score = int(action), score
        return best_action

    def _rollout(self, simulation) -> float:
        while not simulation.is_game_over():
            legal = np.flatnonzero(simulation.available_actions_mask() > 0.0)
            if legal.size == 0:
                break
            simulation.step(int(self.rng.choice(legal)))
        return simulation.score()
