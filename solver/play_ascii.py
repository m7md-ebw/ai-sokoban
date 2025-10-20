import os, time, argparse
from .sokoban_solver import parse_level, State, DIRS, solve_sokoban
from .level_loader import load_boxoban_levels

T = {'floor':' ', 'wall':'#', 'goal':'.', 'box':'$', 'box*':'*', 'player':'@', 'player*':'+'}

def render(level, state):
    grid = [[T['floor'] for _ in range(level.width)] for _ in range(level.height)]
    for (i,j) in level.walls: grid[i][j] = T['wall']
    for (i,j) in level.goals: grid[i][j] = T['goal']
    for (i,j) in state.boxes:
        grid[i][j] = T['box*'] if (i,j) in level.goals else T['box']
    pi,pj = state.player
    grid[pi][pj] = T['player*'] if (pi,pj) in level.goals else T['player']
    return "\n".join("".join(r) for r in grid)

def step(level, state, mv):
    di,dj = DIRS[mv.upper()]
    ni,nj = state.player[0]+di, state.player[1]+dj
    boxes = set(state.boxes)
    if mv.isupper():
        bi,bj = ni+di, nj+dj
        boxes.remove((ni,nj)); boxes.add((bi,bj))
    return State((ni,nj), frozenset(boxes))

def animate(level_lines, moves, fps=10):
    level, state = parse_level(level_lines)
    dt = 1.0 / max(1, fps)
    frames = [render(level, state)]
    for m in moves or "":
        state = step(level, state, m)
        frames.append(render(level, state))
    for f in frames:
        os.system('cls' if os.name=='nt' else 'clear')
        print(f)
        time.sleep(dt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="000.txt")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--method", default="astar", choices=["bfs","dfs","ucs","astar"])
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(here, "..", "boxoban_levels", args.file))
    levels = load_boxoban_levels(path)

    moves = solve_sokoban(levels[args.idx], args.method) or ""
    print("Moves:", moves)
    animate(levels[args.idx], moves, fps=args.fps)
