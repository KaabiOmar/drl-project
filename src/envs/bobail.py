"""Bobail contre un adversaire aleatoire ou heuristique.

Plateau 5x5. L'agent controle le joueur du bas (ligne de depart = ligne 4),
l'adversaire celui du haut (ligne 0). Le bobail (boule neutre) demarre au
centre.

Deroulement d'un tour
---------------------
1. deplacer le bobail d'une case dans une des 8 directions, vers une case vide ;
2. faire glisser un de ses pions en ligne droite dans une des 8 directions,
   jusqu'a buter sur un bord ou une piece (le glissement est maximal).

Exception : le tout premier tour de la partie ne comporte que l'etape 2.

Fin de partie
-------------
* le bobail arrive sur la ligne de depart d'un joueur -> ce joueur GAGNE ;
* un joueur ne peut pas deplacer le bobail au debut de son tour -> il PERD ;
* un joueur ne peut deplacer aucun pion -> il PERD (cas de bord, tres rare).

On gagne donc en RAMENANT le bobail chez soi, pas en le poussant chez l'adverse.
Consequence directe : la ligne de depart est pleine au debut, il faut ouvrir un
trou dans SA PROPRE ligne pour pouvoir y faire entrer le bobail - tout en
empechant l'adversaire d'en faire autant. La regle est isolee dans
`_settle_bobail_position()`.

Encodage
--------
Etat  : 3 plans de 25 cases (pions agent / pions adversaire / bobail) + phase
        encodee en one-hot sur 2 -> 77 valeurs.
Action: 0..7    -> direction du bobail ;
        8..207  -> 8 + case_depart * 8 + direction, glissement d'un pion.
        Soit 208 actions, dont l'immense majorite est masquee a chaque instant.
"""

from __future__ import annotations

import copy

import numpy as np

from src.envs.base import Env

SIZE = 5
CELLS = SIZE * SIZE
EMPTY, AGENT, OPPONENT, BOBAIL = 0, 1, 2, 3

PHASE_BOBAIL, PHASE_PAWN = 0, 1

# 8 directions, indexees de 0 a 7 : N, NE, E, SE, S, SO, O, NO
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
NUM_DIRECTIONS = len(DIRECTIONS)

BOBAIL_ACTIONS = NUM_DIRECTIONS          # actions 0..7
PAWN_ACTIONS = CELLS * NUM_DIRECTIONS    # actions 8..207
NUM_ACTIONS = BOBAIL_ACTIONS + PAWN_ACTIONS

AGENT_HOME_ROW = SIZE - 1
OPPONENT_HOME_ROW = 0


def cell_of(row: int, col: int) -> int:
    return row * SIZE + col


def row_col_of(cell: int) -> tuple[int, int]:
    return divmod(cell, SIZE)


def pawn_action(cell: int, direction: int) -> int:
    """Identifiant de l'action "glisser le pion de `cell` vers `direction`"."""
    return BOBAIL_ACTIONS + cell * NUM_DIRECTIONS + direction


def decode_pawn_action(action: int) -> tuple[int, int]:
    """Inverse de `pawn_action` : renvoie (case de depart, direction)."""
    return divmod(action - BOBAIL_ACTIONS, NUM_DIRECTIONS)


class Bobail(Env):
    name = "bobail"

    def __init__(
        self,
        opponent: str = "random",
        seed: int | None = None,
        max_turns: int = 200,
    ) -> None:
        if opponent not in ("random", "heuristic"):
            raise ValueError("opponent doit valoir 'random' ou 'heuristic'")
        self.opponent = opponent
        self.max_turns = max_turns
        self.rng = np.random.default_rng(seed)
        self.reset()

    # --- interface Env --------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return 3 * CELLS + 2

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def reset(self) -> None:
        self.board = np.zeros(CELLS, dtype=np.int8)
        for col in range(SIZE):
            self.board[cell_of(AGENT_HOME_ROW, col)] = AGENT
            self.board[cell_of(OPPONENT_HOME_ROW, col)] = OPPONENT
        self.bobail = cell_of(SIZE // 2, SIZE // 2)
        self.board[self.bobail] = BOBAIL
        # Le premier tour de la partie n'a pas d'etape "bobail".
        self.phase = PHASE_PAWN
        self.turns = 0
        self._score = 0.0
        self._over = False

    def state_description(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0:CELLS] = (self.board == AGENT).astype(np.float32)
        state[CELLS:2 * CELLS] = (self.board == OPPONENT).astype(np.float32)
        state[2 * CELLS:3 * CELLS] = (self.board == BOBAIL).astype(np.float32)
        state[3 * CELLS + self.phase] = 1.0
        return state

    def available_actions_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if self._over:
            return mask
        if self.phase == PHASE_BOBAIL:
            for direction in self._legal_bobail_directions():
                mask[direction] = 1.0
        else:
            for action in self._legal_pawn_actions(AGENT):
                mask[action] = 1.0
        return mask

    def step(self, action: int) -> None:
        if self._over:
            raise RuntimeError("step() appele sur un episode termine")
        action = int(action)
        if self.available_actions_mask()[action] == 0.0:
            raise ValueError(f"action illegale: {action} (phase={self.phase})")

        if self.phase == PHASE_BOBAIL:
            self._move_bobail(action)
            if self._settle_bobail_position():
                return
            self.phase = PHASE_PAWN
            if not self._legal_pawn_actions(AGENT):
                self._finish(winner=OPPONENT)
            return

        self._slide_pawn(*decode_pawn_action(action))
        self.turns += 1
        if self.turns >= self.max_turns:
            self._over = True  # nul par limite de tours, score inchange
            return

        self._play_opponent_turn()
        self.phase = PHASE_BOBAIL
        if not self._over and not self._legal_bobail_directions():
            # L'agent ne peut pas bouger le bobail au debut de son tour.
            self._finish(winner=OPPONENT)
        elif not self._over and not self._legal_pawn_actions(AGENT):
            self._finish(winner=OPPONENT)

    def is_game_over(self) -> bool:
        return self._over

    def score(self) -> float:
        return self._score

    def clone(self) -> "Bobail":
        new = Bobail.__new__(Bobail)
        new.opponent = self.opponent
        new.max_turns = self.max_turns
        new.rng = copy.deepcopy(self.rng)
        new.board = self.board.copy()
        new.bobail = self.bobail
        new.phase = self.phase
        new.turns = self.turns
        new._score = self._score
        new._over = self._over
        return new

    def render(self) -> str:
        glyphs = {EMPTY: ".", AGENT: "A", OPPONENT: "O", BOBAIL: "B"}
        rows = [
            "".join(glyphs[int(self.board[cell_of(r, c)])] for c in range(SIZE))
            for r in range(SIZE)
        ]
        phase = "bobail" if self.phase == PHASE_BOBAIL else "pion"
        return "\n".join(rows) + f"\n(tour {self.turns}, phase {phase})"

    # --- regles ---------------------------------------------------------------

    def _legal_bobail_directions(self) -> list[int]:
        row, col = row_col_of(self.bobail)
        legal = []
        for direction, (d_row, d_col) in enumerate(DIRECTIONS):
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < SIZE and 0 <= new_col < SIZE:
                if self.board[cell_of(new_row, new_col)] == EMPTY:
                    legal.append(direction)
        return legal

    def _legal_pawn_actions(self, player: int) -> list[int]:
        legal = []
        for cell in np.flatnonzero(self.board == player):
            for direction in range(NUM_DIRECTIONS):
                if self.slide_target(int(cell), direction) is not None:
                    legal.append(pawn_action(int(cell), direction))
        return legal

    def slide_target(self, cell: int, direction: int) -> int | None:
        """Case d'arrivee d'un glissement maximal, ou None si le pion est bloque.

        Publique : l'interface graphique s'en sert pour montrer les destinations
        legales d'un pion selectionne.
        """
        d_row, d_col = DIRECTIONS[direction]
        row, col = row_col_of(cell)
        target = None
        while True:
            row, col = row + d_row, col + d_col
            if not (0 <= row < SIZE and 0 <= col < SIZE):
                break
            if self.board[cell_of(row, col)] != EMPTY:
                break
            target = cell_of(row, col)
        return target

    def _move_bobail(self, direction: int) -> None:
        row, col = row_col_of(self.bobail)
        d_row, d_col = DIRECTIONS[direction]
        destination = cell_of(row + d_row, col + d_col)
        self.board[self.bobail] = EMPTY
        self.board[destination] = BOBAIL
        self.bobail = destination

    def _slide_pawn(self, cell: int, direction: int) -> None:
        target = self.slide_target(cell, direction)
        if target is None:
            raise ValueError(f"glissement impossible depuis {cell} vers {direction}")
        self.board[target] = self.board[cell]
        self.board[cell] = EMPTY

    def _settle_bobail_position(self) -> bool:
        """Applique la regle de victoire liee a la ligne atteinte par le bobail."""
        row, _ = row_col_of(self.bobail)
        if row == AGENT_HOME_ROW:
            self._finish(winner=AGENT)
        elif row == OPPONENT_HOME_ROW:
            self._finish(winner=OPPONENT)
        return self._over

    def _finish(self, winner: int) -> None:
        self._score += 1.0 if winner == AGENT else -1.0
        self._over = True

    # --- adversaire -----------------------------------------------------------

    def _play_opponent_turn(self) -> None:
        directions = self._legal_bobail_directions()
        if not directions:
            self._finish(winner=AGENT)
            return
        self._move_bobail(self._choose_opponent_bobail(directions))
        if self._settle_bobail_position():
            return

        actions = self._legal_pawn_actions(OPPONENT)
        if not actions:
            self._finish(winner=AGENT)
            return
        self._slide_pawn(*decode_pawn_action(self._choose_opponent_pawn(actions)))

    def _choose_opponent_bobail(self, directions: list[int]) -> int:
        """L'adversaire gagne en ramenant le bobail sur SA ligne, la ligne 0."""
        if self.opponent == "random":
            return int(self.rng.choice(directions))
        best = min(DIRECTIONS[d][0] for d in directions)
        greedy = [d for d in directions if DIRECTIONS[d][0] == best]
        return int(self.rng.choice(greedy))

    def _choose_opponent_pawn(self, actions: list[int]) -> int:
        """Heuristique volontairement simple : se rapprocher du bobail."""
        if self.opponent == "random":
            return int(self.rng.choice(actions))
        bobail_row, bobail_col = row_col_of(self.bobail)

        def distance(action: int) -> int:
            cell, direction = decode_pawn_action(action)
            target = self.slide_target(cell, direction)
            row, col = row_col_of(target)
            return max(abs(row - bobail_row), abs(col - bobail_col))

        best = min(distance(a) for a in actions)
        greedy = [a for a in actions if distance(a) == best]
        return int(self.rng.choice(greedy))
