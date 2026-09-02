"""Regles du Bobail.

Le moteur est la brique dont tout depend : une regle fausse ici invalide
silencieusement les 11 series d'entrainement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.bobail import (
    AGENT,
    OPPONENT_HOME_ROW,
    row_col_of,
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


def surround_bobail(env, filler: int = OPPONENT) -> None:
    """Bouche les 8 cases autour du bobail."""
    row, col = row_col_of(env.bobail)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if (d_row, d_col) != (0, 0):
                env.board[cell_of(row + d_row, col + d_col)] = filler


def test_agent_loses_when_it_cannot_move_the_bobail() -> None:
    """Bobail enferme au debut du tour de l'agent : l'agent perd."""
    env = Bobail(seed=0)
    env.reset()
    surround_bobail(env)
    assert env._legal_bobail_directions() == []

    env._start_agent_turn()
    assert env.is_game_over()
    assert env.score() == pytest.approx(-1.0)


def test_agent_wins_by_trapping_the_bobail_with_its_last_pawn_move() -> None:
    """Enfermer la boule est la seconde facon de gagner.

    Position construite : le bobail en (2,0) n'a plus qu'une case libre, (3,0).
    Le pion de l'agent en (4,0) glisse vers le nord et la bouche. L'adversaire
    ouvre alors son tour sans pouvoir deplacer le bobail : il perd.
    """
    env = Bobail(seed=0)
    env.reset()
    env.board[:] = EMPTY

    env.bobail = cell_of(2, 0)
    env.board[env.bobail] = BOBAIL
    for row, col in ((1, 0), (1, 1), (2, 1), (3, 1)):
        env.board[cell_of(row, col)] = OPPONENT
    env.board[cell_of(4, 0)] = AGENT
    env.phase = PHASE_PAWN

    north = 0
    env.step(pawn_action(cell_of(4, 0), north))

    assert env.board[cell_of(3, 0)] == AGENT, "le pion doit boucher la derniere case"
    assert env.is_game_over()
    assert env.score() == pytest.approx(1.0)


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


# --- heuristique de l'adversaire ------------------------------------------


def place_bobail(env, row: int, col: int) -> None:
    """Deplace le bobail sur une case donnee, en liberant son voisinage."""
    env.board[env.bobail] = EMPTY
    env.bobail = cell_of(row, col)
    env.board[env.bobail] = BOBAIL


def test_heuristic_opponent_takes_an_immediate_win() -> None:
    """Priorite 1 : un coup qui amene le bobail sur la ligne 0 gagne, il faut le jouer."""
    env = Bobail(opponent="heuristic", seed=0)
    env.reset()
    place_bobail(env, 1, 2)
    env.board[cell_of(OPPONENT_HOME_ROW, 2)] = EMPTY  # une case liberee sur sa ligne

    directions = env._legal_bobail_directions()
    choisi = env._choose_opponent_bobail(directions)

    assert env._bobail_target_row(choisi) == OPPONENT_HOME_ROW, "l'adversaire doit conclure"


def test_heuristic_opponent_refuses_to_hand_the_win_over() -> None:
    """Priorite 2 : ne jamais poser le bobail sur la ligne de l'agent."""
    env = Bobail(opponent="heuristic", seed=0)
    env.reset()
    place_bobail(env, 3, 2)
    env.board[cell_of(AGENT_HOME_ROW, 2)] = EMPTY  # le coup suicidaire est disponible

    directions = env._legal_bobail_directions()
    suicides = [d for d in directions if env._bobail_target_row(d) == AGENT_HOME_ROW]
    assert suicides, "le test n'a de sens que si un coup suicidaire est legal"

    for _ in range(20):  # le choix comporte un tirage aleatoire entre ex aequo
        assert env._choose_opponent_bobail(directions) not in suicides


def test_heuristic_opponent_otherwise_advances_toward_its_row() -> None:
    """Priorite 3 : a defaut, se rapprocher de la ligne 0."""
    env = Bobail(opponent="heuristic", seed=0)
    env.reset()
    place_bobail(env, 2, 2)

    directions = env._legal_bobail_directions()
    choisi = env._choose_opponent_bobail(directions)
    lignes = [env._bobail_target_row(d) for d in directions]

    assert env._bobail_target_row(choisi) == min(lignes)
