"""Regles du Bobail.

Le moteur est la brique dont tout depend : une regle fausse ici invalide
silencieusement les 11 series d'entrainement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.bobail import (
    AGENT,
    AGENT_HOME_ROW,
    BOBAIL,
    EMPTY,
    NUM_ACTIONS,
    OPPONENT,
    OPPONENT_HOME_ROW,
    PHASE_BOBAIL,
    PHASE_PAWN,
    Bobail,
    cell_of,
    decode_pawn_action,
    pawn_action,
)


def test_initial_layout() -> None:
    env = Bobail(seed=0)
    assert all(env.board[cell_of(AGENT_HOME_ROW, c)] == AGENT for c in range(5))
    assert all(env.board[cell_of(OPPONENT_HOME_ROW, c)] == OPPONENT for c in range(5))
    assert env.board[cell_of(2, 2)] == BOBAIL
    assert env.bobail == cell_of(2, 2)


def test_first_turn_skips_the_bobail_move() -> None:
    env = Bobail(seed=0)
    assert env.phase == PHASE_PAWN
    mask = env.available_actions_mask()
    assert mask[:8].sum() == 0, "aucun deplacement de bobail au premier tour"
    assert mask[8:].sum() > 0


def test_action_encoding_round_trip() -> None:
    for cell in range(25):
        for direction in range(8):
            assert decode_pawn_action(pawn_action(cell, direction)) == (cell, direction)


def test_pawn_slide_is_maximal() -> None:
    env = Bobail(seed=0)
    # Le pion de la case (4,0) glisse vers le nord : il doit s'arreter juste
    # sous la ligne adverse, en (1,0), et non avancer d'une seule case.
    north = 0
    env.step(pawn_action(cell_of(AGENT_HOME_ROW, 0), north))
    assert env.board[cell_of(1, 0)] == AGENT
    assert env.board[cell_of(AGENT_HOME_ROW, 0)] == EMPTY


def test_bobail_reaching_opponent_home_row_loses() -> None:
    env = Bobail(seed=0)
    env.reset()
    # Position construite a la main : bobail en (1,2), a une case de la ligne 0.
    # On gagne en ramenant le bobail CHEZ SOI : l'amener chez l'adversaire, meme
    # volontairement, le fait gagner lui.
    env.board[env.bobail] = EMPTY
    env.bobail = cell_of(1, 2)
    env.board[env.bobail] = BOBAIL
    env.board[cell_of(OPPONENT_HOME_ROW, 2)] = EMPTY
    env.phase = PHASE_BOBAIL

    north = 0
    env.step(north)
    assert env.is_game_over()
    assert env.score() == pytest.approx(-1.0)


def test_bobail_reaching_own_home_row_wins() -> None:
    env = Bobail(seed=0)
    env.reset()
    env.board[env.bobail] = EMPTY
    env.bobail = cell_of(3, 2)
    env.board[env.bobail] = BOBAIL
    env.board[cell_of(AGENT_HOME_ROW, 2)] = EMPTY  # on libere la case d'arrivee
    env.phase = PHASE_BOBAIL

    south = 4
    env.step(south)
    assert env.is_game_over()
    assert env.score() == pytest.approx(1.0)


def test_blocked_bobail_loses_for_the_player_to_move() -> None:
    env = Bobail(seed=0)
    env.reset()
    env.phase = PHASE_BOBAIL
    # On entoure le bobail de pions : plus aucun deplacement possible.
    row, col = 2, 2
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if (d_row, d_col) != (0, 0):
                env.board[cell_of(row + d_row, col + d_col)] = OPPONENT
    assert env._legal_bobail_directions() == []
    assert env.available_actions_mask().sum() == 0


def test_mask_only_exposes_legal_actions() -> None:
    rng = np.random.default_rng(0)
    env = Bobail(seed=0)
    for _ in range(50):
        env.reset()
        while not env.is_game_over():
            mask = env.available_actions_mask()
            assert mask.shape == (NUM_ACTIONS,)
            legal = np.flatnonzero(mask > 0.0)
            if legal.size == 0:
                break
            if env.phase == PHASE_BOBAIL:
                assert (legal < 8).all(), "phase bobail: seules les 8 directions sont legales"
            else:
                assert (legal >= 8).all(), "phase pion: aucune action de bobail"
            env.step(int(rng.choice(legal)))


def test_heuristic_opponent_plays_legally() -> None:
    rng = np.random.default_rng(0)
    env = Bobail(opponent="heuristic", seed=0)
    for _ in range(20):
        env.reset()
        while not env.is_game_over():
            legal = env.available_actions()
            if legal.size == 0:
                break
            env.step(int(rng.choice(legal)))
        assert env.is_game_over()
