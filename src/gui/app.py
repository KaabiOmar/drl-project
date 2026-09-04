"""Interface graphique : regarder jouer un agent, ou jouer soi-meme.

Livrable impose par le sujet ("presenter une interface graphique permettant de
regarder jouer chaque agent et egalement de mettre a disposition un agent
humain").

Usage :
    python -m src.gui.app --env bobail --agent mcts_uct
    python -m src.gui.app --env bobail --agent human
    python -m src.gui.app --env bobail --agent dqn --model runs/.../model_100000.pt

Commandes : ESPACE met en pause / reprend, N joue un coup, R relance une partie,
FLECHES pour les environnements sans plateau, ECHAP quitte.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents import AGENT_REGISTRY, make_agent
from src.envs import make_env
from src.gui.views import BACKGROUND, HIGHLIGHT, SELECTION, TEXT, VIEWS

MARGIN = 32
PANEL_HEIGHT = 110


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualisation des agents")
    parser.add_argument("--env", default="bobail")
    parser.add_argument("--agent", default="human", help="human | " + " | ".join(sorted(AGENT_REGISTRY)))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=int, default=400, help="ms entre deux coups de l'agent")
    return parser.parse_args()


def build_env(args: argparse.Namespace):
    kwargs = {"seed": args.seed} if args.env in ("tictactoe", "bobail") else {}
    if args.env == "bobail":
        kwargs["opponent"] = args.opponent
    return make_env(args.env, **kwargs)


def main() -> None:
    import pygame

    args = parse_args()
    env = build_env(args)
    view = VIEWS[args.env]()

    agent = None
    if args.agent != "human":
        agent = make_agent(args.agent, env, seed=args.seed)
        if args.model is not None:
            agent.load(args.model)

    rows, cols = view.grid_shape(env)
    width = cols * view.cell_size + 2 * MARGIN
    height = rows * view.cell_size + 2 * MARGIN + PANEL_HEIGHT

    pygame.init()
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"DRL - {args.env} - {args.agent}")
    font = pygame.font.SysFont("menlo,monospace", 15)
    clock = pygame.time.Clock()

    selection: int | None = None
    paused = False
    last_move = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key == "escape":
                    running = False
                elif key == "r":
                    env.reset()
                    selection = None
                elif key == "space":
                    paused = not paused
                elif key == "n" and agent is not None and not env.is_game_over():
                    play_agent_move(env, agent)
                elif agent is None and not env.is_game_over():
                    action = view.action_from_key(env, key)
                    if action is not None and env.available_actions_mask()[action] > 0:
                        env.step(action)
            elif event.type == pygame.MOUSEBUTTONDOWN and agent is None and not env.is_game_over():
                cell = cell_at(event.pos, view, rows, cols)
                if cell is not None:
                    action, selection = view.action_from_click(env, cell, selection)
                    if action is not None and env.available_actions_mask()[action] > 0:
                        env.step(action)

        now = pygame.time.get_ticks()
        if agent is not None and not paused and not env.is_game_over() and now - last_move > args.delay:
            play_agent_move(env, agent)
            last_move = now

        surface.fill(BACKGROUND)
        draw_hints(pygame, surface, view, env, selection, cols)
        view.draw(pygame, surface, env, (MARGIN, MARGIN))
        draw_panel(pygame, surface, font, env, args, paused, height)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def draw_hints(pygame, surface, view, env, selection: int | None, cols: int) -> None:
    """Marque la piece selectionnee et les destinations legales (mode humain)."""
    for cell in view.highlights(env, selection):
        row, col = divmod(cell, cols)
        rect = pygame.Rect(
            MARGIN + col * view.cell_size,
            MARGIN + row * view.cell_size,
            view.cell_size,
            view.cell_size,
        )
        pygame.draw.rect(surface, HIGHLIGHT, rect.inflate(-16, -16), width=2, border_radius=6)
    if selection is not None:
        row, col = divmod(selection, cols)
        view._ring(pygame, surface, (MARGIN, MARGIN), row, col, SELECTION)


def play_agent_move(env, agent) -> None:
    mask = env.available_actions_mask()
    if mask.sum() == 0:
        return
    # Mode glouton : on regarde jouer la POLITIQUE OBTENUE, sans exploration.
    action = agent.act(env.state_description(), mask, env=env, greedy=True)
    env.step(action)


def cell_at(position, view, rows: int, cols: int) -> int | None:
    x, y = position
    col = (x - MARGIN) // view.cell_size
    row = (y - MARGIN) // view.cell_size
    if 0 <= row < rows and 0 <= col < cols:
        return int(row * cols + col)
    return None


def draw_panel(pygame, surface, font, env, args, paused: bool, height: int) -> None:
    status = "terminee" if env.is_game_over() else ("en pause" if paused else "en cours")
    lines = [
        f"env {args.env}   agent {args.agent}   partie {status}",
        f"score {env.score():+.1f}",
        "espace pause   n coup suivant   r rejouer   echap quitter",
    ]
    for index, line in enumerate(lines):
        surface.blit(font.render(line, True, TEXT), (MARGIN, height - PANEL_HEIGHT + 16 + index * 24))


if __name__ == "__main__":
    main()
