"""Rendu et saisie, un adaptateur par environnement.

Separer la vue du moteur permet d'ajouter un environnement sans toucher a la
boucle d'affichage, et surtout de garder `src/envs` totalement independant de
pygame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.envs.bobail import (
    AGENT,
    BOBAIL,
    EMPTY,
    DIRECTIONS,
    OPPONENT,
    PHASE_BOBAIL,
    SIZE,
    cell_of,
    pawn_action,
    row_col_of,
)

BACKGROUND = (24, 26, 32)
GRID = (60, 64, 76)
TEXT = (232, 234, 240)
AGENT_COLOR = (86, 156, 246)
OPPONENT_COLOR = (232, 106, 96)
BOBAIL_COLOR = (240, 214, 120)
HIGHLIGHT = (120, 200, 140)
SELECTION = (250, 250, 250)


class BoardView(ABC):
    """Adaptateur d'affichage et de saisie pour un environnement."""

    cell_size = 96

    @abstractmethod
    def grid_shape(self, env) -> tuple[int, int]:
        """(lignes, colonnes) de la grille affichee."""

    @abstractmethod
    def draw(self, pygame, surface, env, origin: tuple[int, int]) -> None:
        """Dessine le plateau a partir de `origin`."""

    def action_from_click(self, env, cell: int, selection: int | None) -> tuple[int | None, int | None]:
        """Traduit un clic en action.

        Renvoie `(action, nouvelle_selection)`. `action=None` signifie que le
        clic n'a fait que selectionner une piece.
        """
        return None, None

    def action_from_key(self, env, key_name: str) -> int | None:
        """Traduit une touche en action, pour les environnements sans plateau."""
        return None

    def highlights(self, env, selection: int | None) -> list[int]:
        """Cases a marquer comme destinations legales, pour le mode humain."""
        return []


class GridLikeView(BoardView):
    """Vue commune aux environnements en grille (une case = une position)."""

    def _draw_grid(self, pygame, surface, rows: int, cols: int, origin: tuple[int, int]) -> None:
        for row in range(rows):
            for col in range(cols):
                rect = pygame.Rect(
                    origin[0] + col * self.cell_size,
                    origin[1] + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(surface, GRID, rect, width=1)

    def _ring(self, pygame, surface, origin, row, col, color, radius_ratio=0.42) -> None:
        center = (
            origin[0] + int((col + 0.5) * self.cell_size),
            origin[1] + int((row + 0.5) * self.cell_size),
        )
        pygame.draw.circle(surface, color, center, int(self.cell_size * radius_ratio), width=2)

    def _circle(self, pygame, surface, origin, row, col, color, radius_ratio=0.34) -> None:
        center = (
            origin[0] + int((col + 0.5) * self.cell_size),
            origin[1] + int((row + 0.5) * self.cell_size),
        )
        pygame.draw.circle(surface, color, center, int(self.cell_size * radius_ratio))


class LineWorldView(GridLikeView):
    def grid_shape(self, env) -> tuple[int, int]:
        return 1, env.size

    def draw(self, pygame, surface, env, origin) -> None:
        self._draw_grid(pygame, surface, 1, env.size, origin)
        self._circle(pygame, surface, origin, 0, 0, OPPONENT_COLOR, 0.18)
        self._circle(pygame, surface, origin, 0, env.size - 1, HIGHLIGHT, 0.18)
        self._circle(pygame, surface, origin, 0, env.position, AGENT_COLOR)

    def action_from_key(self, env, key_name: str) -> int | None:
        return {"left": 0, "right": 1}.get(key_name)


class GridWorldView(GridLikeView):
    def grid_shape(self, env) -> tuple[int, int]:
        return env.rows, env.cols

    def draw(self, pygame, surface, env, origin) -> None:
        self._draw_grid(pygame, surface, env.rows, env.cols, origin)
        self._circle(pygame, surface, origin, *env.goal, HIGHLIGHT, 0.18)
        self._circle(pygame, surface, origin, *env.trap, OPPONENT_COLOR, 0.18)
        self._circle(pygame, surface, origin, env.row, env.col, AGENT_COLOR)

    def action_from_key(self, env, key_name: str) -> int | None:
        return {"up": 0, "down": 1, "left": 2, "right": 3}.get(key_name)


class TicTacToeView(GridLikeView):
    def grid_shape(self, env) -> tuple[int, int]:
        return 3, 3

    def draw(self, pygame, surface, env, origin) -> None:
        self._draw_grid(pygame, surface, 3, 3, origin)
        for cell in range(9):
            row, col = divmod(cell, 3)
            value = int(env.board[cell])
            if value == 1:
                self._circle(pygame, surface, origin, row, col, AGENT_COLOR)
            elif value == 2:
                self._circle(pygame, surface, origin, row, col, OPPONENT_COLOR)

    def action_from_click(self, env, cell: int, selection: int | None):
        return cell, None


class BobailView(GridLikeView):
    def grid_shape(self, env) -> tuple[int, int]:
        return SIZE, SIZE

    def draw(self, pygame, surface, env, origin) -> None:
        self._draw_grid(pygame, surface, SIZE, SIZE, origin)
        for cell in range(SIZE * SIZE):
            row, col = row_col_of(cell)
            value = int(env.board[cell])
            if value == AGENT:
                self._circle(pygame, surface, origin, row, col, AGENT_COLOR)
            elif value == OPPONENT:
                self._circle(pygame, surface, origin, row, col, OPPONENT_COLOR)
            elif value == BOBAIL:
                self._circle(pygame, surface, origin, row, col, BOBAIL_COLOR, 0.22)

    def action_from_click(self, env, cell: int, selection: int | None):
        """Phase bobail : cliquer une case adjacente. Phase pion : pion puis destination."""
        if env.phase == PHASE_BOBAIL:
            bobail_row, bobail_col = row_col_of(env.bobail)
            row, col = row_col_of(cell)
            for direction, (d_row, d_col) in enumerate(DIRECTIONS):
                if (bobail_row + d_row, bobail_col + d_col) == (row, col):
                    return direction, None
            return None, None

        if selection is None:
            return (None, cell) if env.board[cell] == AGENT else (None, None)

        start_row, start_col = row_col_of(selection)
        row, col = row_col_of(cell)
        for direction in range(len(DIRECTIONS)):
            if env.slide_target(selection, direction) == cell:
                return pawn_action(selection, direction), None
        # Clic ailleurs : on change de pion selectionne, ou on annule.
        return (None, cell) if env.board[cell] == AGENT else (None, None)

    def highlights(self, env, selection: int | None) -> list[int]:
        if env.phase == PHASE_BOBAIL:
            row, col = row_col_of(env.bobail)
            cells = []
            for d_row, d_col in DIRECTIONS:
                new_row, new_col = row + d_row, col + d_col
                if 0 <= new_row < SIZE and 0 <= new_col < SIZE:
                    if env.board[cell_of(new_row, new_col)] == EMPTY:
                        cells.append(cell_of(new_row, new_col))
            return cells
        if selection is None:
            return []
        targets = [env.slide_target(selection, d) for d in range(len(DIRECTIONS))]
        return [cell for cell in targets if cell is not None]


VIEWS: dict[str, type[BoardView]] = {
    "line_world": LineWorldView,
    "grid_world": GridWorldView,
    "tictactoe": TicTacToeView,
    "bobail": BobailView,
}
