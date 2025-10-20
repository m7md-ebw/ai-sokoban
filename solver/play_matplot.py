import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from .sokoban_solver import parse_level, State, DIRS, solve_sokoban
from .level_loader import load_boxoban_levels

def draw(ax, level, state):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, level.width-0.5)
    ax.set_ylim(level.height-0.5, -0.5)
    ax.axis('off')
    for i in range(level.height):
        for j in range(level.width):
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, linewidth=0.3))
    for (i,j) in level.walls:
        ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1))
    for (i,j) in level.goals:
        ax.add_patch(Circle((j, i), 0.15))
    for (i,j) in state.boxes:
        ax.add_patch(Rectangle((j-0.35, i-0.35), 0.7, 0.7))
        if (i,j) in level.goals:
            ax.add_patch(Circle((j, i), 0.12))
    pi,pj = state.player
    ax.add_patch(Circle((pj, pi), 0.3))

def step(level, state, mv):
    di,dj = DIRS[mv.upper()]
    ni,nj = state.player[0]+di, state.player[1]+dj
    boxes = set(state.boxes)
    if mv.isupper():
        bi,bj = ni+di, nj+dj
        boxes.remove((ni,nj)); boxes.add((bi,bj))
    return State((ni,nj), frozenset(boxes))

def playback(level_lines, moves, delay=0.08):
    level, state = parse_level(level_lines)
    fig, ax = plt.subplots()
    draw(ax, level, state); plt.pause(0.001)
    for m in moves or "":
        state = step(level, state, m)
        draw(ax, level, state); plt.pause(delay)
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="000.txt")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--method", default="astar", choices=["bfs","dfs","ucs","astar"])
    ap.add_argument("--delay", type=float, default=0.08)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(here, "..", "boxoban_levels", args.file))
    levels = load_boxoban_levels(path)

    moves = solve_sokoban(levels[args.idx], args.method) or ""
    print("Moves:", moves)
    playback(levels[args.idx], moves, delay=args.delay)
