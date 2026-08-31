"""Conformite des quatre environnements au contrat `Env`.

Ces tests tournent sur les quatre environnements en parametrique : toute
violation du contrat casse immediatement tous les agents.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs import ENV_REGISTRY, make_env

ENV_NAMES = sorted(ENV_REGISTRY)


def build(name: str):
    kwargs = {"seed": 0} if name in ("tictactoe", "bobail") else {}
    return make_env(name, **kwargs)


@pytest.mark.parametrize("name", ENV_NAMES)
def test_state_description_matches_declared_dimension(name: str) -> None:
    env = build(name)
    state = env.state_description()
    assert state.shape == (env.state_dim,)
    assert state.dtype == np.float32


@pytest.mark.parametrize("name", ENV_NAMES)
def test_mask_matches_declared_action_space(name: str) -> None:
    env = build(name)
    mask = env.available_actions_mask()
    assert mask.shape == (env.num_actions,)
    assert mask.sum() > 0, "l'etat initial doit offrir au moins une action legale"


@pytest.mark.parametrize("name", ENV_NAMES)
def test_random_playthrough_terminates(name: str) -> None:
    rng = np.random.default_rng(0)
    env = build(name)
    for _ in range(20):
        env.reset()
        steps = 0
        while not env.is_game_over() and steps < 5_000:
            legal = env.available_actions()
            assert legal.size > 0
            env.step(int(rng.choice(legal)))
            steps += 1
        assert env.is_game_over(), "la partie doit se terminer"


@pytest.mark.parametrize("name", ENV_NAMES)
def test_clone_is_independent(name: str) -> None:
    rng = np.random.default_rng(1)
    env = build(name)
    env.reset()
    snapshot = env.state_description().copy()

    copy = env.clone()
    while not copy.is_game_over():
        legal = copy.available_actions()
        if legal.size == 0:
            break
        copy.step(int(rng.choice(legal)))

    assert np.array_equal(env.state_description(), snapshot), "le clone a mute l'original"
    assert not env.is_game_over()


@pytest.mark.parametrize("name", ENV_NAMES)
def test_step_refuses_illegal_action(name: str) -> None:
    env = build(name)
    env.reset()
    mask = env.available_actions_mask()
    illegal = np.flatnonzero(mask == 0.0)
    if illegal.size == 0:
        pytest.skip(f"{name} n'a aucune action illegale a l'etat initial")
    with pytest.raises((ValueError, IndexError)):
        env.step(int(illegal[0]))
