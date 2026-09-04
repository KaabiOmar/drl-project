"""Faire tenir le role de l'adversaire par un agent entraine.

Nos agents apprennent tous a jouer UN SEUL cote : celui que l'environnement
appelle AGENT. Pour qu'un modele joue l'autre camp, il faut donc lui presenter
le plateau vu depuis ce camp, puis retraduire le coup qu'il renvoie.

Deux transformations, une par environnement :

* TicTacToe : il suffit d'echanger les deux plans de pieces. Les cases ne
  bougent pas, donc l'action n'a pas besoin d'etre retraduite.
* Bobail : les deux camps n'ont pas la meme ligne de depart. Il faut donc
  echanger les plans ET retourner le plateau verticalement, puis appliquer le
  meme retournement a l'action renvoyee.

Le masque est construit a partir des coups legaux que l'environnement nous
passe, transposes dans le repere de l'agent. On ne recalcule jamais la legalite
dans le repere retourne : la transformation etant bijective et involutive, la
correspondance est exacte.
"""

from __future__ import annotations

import numpy as np

from src.envs.bobail import (
    AGENT,
    BOBAIL,
    CELLS,
    DIRECTIONS,
    OPPONENT,
    PHASE_BOBAIL,
    SIZE,
    cell_of,
    row_col_of,
)

# Retournement vertical des huit directions : N <-> S, NE <-> SE, SO <-> NO,
# E et O inchanges. L'application est sa propre inverse.
MIROIR_DIRECTION = [4, 3, 2, 1, 0, 7, 6, 5]


def _verifier_miroir() -> None:
    """Garde-fou : le retournement doit vraiment negrer le deplacement vertical."""
    for d, cible in enumerate(MIROIR_DIRECTION):
        d_row, d_col = DIRECTIONS[d]
        c_row, c_col = DIRECTIONS[cible]
        assert (c_row, c_col) == (-d_row, d_col), f"direction {d} mal retournee"


_verifier_miroir()


def miroir_case(cellule: int) -> int:
    ligne, colonne = row_col_of(cellule)
    return cell_of(SIZE - 1 - ligne, colonne)


def miroir_action_bobail(action: int) -> int:
    """Traduit une action entre le repere reel et le repere retourne.

    Involutive : l'appliquer deux fois redonne l'action de depart.
    """
    if action < len(DIRECTIONS):
        return MIROIR_DIRECTION[action]
    cellule, direction = divmod(action - len(DIRECTIONS), len(DIRECTIONS))
    return len(DIRECTIONS) + miroir_case(cellule) * len(DIRECTIONS) + MIROIR_DIRECTION[direction]


# --- TicTacToe ------------------------------------------------------------


def adversaire_tictactoe(agent):
    """Renvoie une politique adverse pilotee par `agent`, pour TicTacToe."""
    echange = {0: 0, 1: 2, 2: 1}  # vide reste vide, les deux camps s'echangent

    def jouer(env, cases_libres):
        etat = np.zeros(27, dtype=np.float32)
        for cellule in range(9):
            etat[echange[int(env.board[cellule])] * 9 + cellule] = 1.0
        masque = np.zeros(9, dtype=np.float32)
        masque[np.asarray(cases_libres, dtype=int)] = 1.0
        return agent.act(etat, masque, env=None, greedy=True)

    return jouer


# --- Bobail ---------------------------------------------------------------


def adversaire_bobail(agent):
    """Renvoie une politique adverse pilotee par `agent`, pour Bobail."""

    def jouer(env, phase, coups_legaux):
        etat = np.zeros(3 * CELLS + 2, dtype=np.float32)
        for cellule in range(CELLS):
            piece = int(env.board[cellule])
            vu = miroir_case(cellule)
            if piece == OPPONENT:
                etat[vu] = 1.0                    # « mes » pions
            elif piece == AGENT:
                etat[CELLS + vu] = 1.0            # « ses » pions
            elif piece == BOBAIL:
                etat[2 * CELLS + vu] = 1.0
        etat[3 * CELLS + phase] = 1.0

        masque = np.zeros(env.num_actions, dtype=np.float32)
        correspondance = {}
        for reel in coups_legaux:
            vu = miroir_action_bobail(int(reel))
            masque[vu] = 1.0
            correspondance[vu] = int(reel)

        choisi = agent.act(etat, masque, env=None, greedy=True)
        return correspondance[int(choisi)]

    return jouer


FABRIQUES = {
    "tictactoe": adversaire_tictactoe,
    "bobail": adversaire_bobail,
}


def brancher(env_name: str, env, agent) -> None:
    """Installe `agent` comme adversaire de `env`.

    Leve une erreur explicite sur les environnements a un seul joueur, ou il n'y
    a pas d'adversaire a remplacer.
    """
    if env_name not in FABRIQUES:
        raise ValueError(
            f"{env_name} n'a pas d'adversaire : jouer contre un agent n'a de sens "
            f"que sur {' ou '.join(sorted(FABRIQUES))}"
        )
    env.external_opponent = FABRIQUES[env_name](agent)
