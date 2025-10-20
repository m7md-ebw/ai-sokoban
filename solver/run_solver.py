import os, argparse
from .sokoban_solver import solve_sokoban
from .level_loader import load_boxoban_levels

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="000.txt", help="level file inside boxoban_levels/")
    ap.add_argument("--idx", type=int, default=0, help="which level index to solve")
    ap.add_argument("--method", default="astar", choices=["bfs","dfs","ucs","astar"])
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(here, "..", "boxoban_levels", args.file))
    levels = load_boxoban_levels(path)

    moves = solve_sokoban(levels[args.idx], args.method)
    print(f"Level {args.idx} ({args.method})")
    print("Moves:", moves if moves else "No solution found")
