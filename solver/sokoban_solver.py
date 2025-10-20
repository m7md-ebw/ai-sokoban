import heapq
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Set, FrozenSet, Optional, Iterator

Coord = Tuple[int, int]

@dataclass(frozen=True)
class Level:
    walls: FrozenSet[Coord]
    goals: FrozenSet[Coord]
    width: int
    height: int

@dataclass(frozen=True)
class State:
    player: Coord
    boxes: FrozenSet[Coord]

# Directions
DIRS = {
    'U': (-1, 0),
    'D': ( 1, 0),
    'L': ( 0,-1),
    'R': ( 0, 1),
}
DIR_LOWER = {'U':'u', 'D':'d', 'L':'l', 'R':'r'}
DIR_UPPER = {'U':'U', 'D':'D', 'L':'L', 'R':'R'}

# ---------- Level parsing ----------

def parse_level(lines: Iterable[str]) -> Tuple[Level, State]:
    """
    Accepts an iterable of strings (lines of a single level).
    Supports both Boxoban symbols and the older symbols.
    """
    grid = [list(r.rstrip('\n')) for r in lines]
    if not grid:
        raise ValueError("Empty level.")

    h = len(grid)
    w = max(len(r) for r in grid)
    # pad ragged rows
    for r in grid:
        r += [' '] * (w - len(r))

    walls: Set[Coord] = set()
    goals: Set[Coord] = set()
    boxes: Set[Coord] = set()
    player: Optional[Coord] = None

    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            if c == '#':
                walls.add((i, j))
            elif c == '.':
                goals.add((i, j))
            # Boxoban symbols
            elif c == '$':
                boxes.add((i, j))
            elif c == '*':  # box on goal
                boxes.add((i, j)); goals.add((i, j))
            elif c == '@':  # player
                player = (i, j)
            elif c == '+':  # player on goal
                player = (i, j); goals.add((i, j))
            # Older variant symbols
            elif c == 'B':
                boxes.add((i, j))
            elif c == 'X':
                boxes.add((i, j)); goals.add((i, j))
            elif c == '&':
                player = (i, j)

    if player is None:
        raise ValueError("No player found in level (@, +, or &).")
    if not boxes:
        raise ValueError("No boxes found ($, *, B, or X).")

    lvl = Level(frozenset(walls), frozenset(goals), w, h)
    st = State(player, frozenset(boxes))
    return lvl, st

# ---------- Helpers ----------

def is_goal_state(state: State, level: Level) -> bool:
    return state.boxes.issubset(level.goals) and len(state.boxes) == len(level.goals)

def inside(i: int, j: int, level: Level) -> bool:
    return 0 <= i < level.height and 0 <= j < level.width

def corner_deadlock(cell: Coord, level: Level, goals: FrozenSet[Coord]) -> bool:
    """
    Simple deadlock: a box pushed into a strict corner (two walls) that is NOT a goal.
    """
    if cell in goals:
        return False
    i, j = cell
    up    = (i-1, j)
    down  = (i+1, j)
    left  = (i, j-1)
    right = (i, j+1)
    def blocked(c: Coord) -> bool:
        return (c in level.walls) or (not inside(c[0], c[1], level))
    if blocked(up) and blocked(left):   return True
    if blocked(up) and blocked(right):  return True
    if blocked(down) and blocked(left): return True
    if blocked(down) and blocked(right):return True
    return False

def any_simple_deadlock(new_boxes: FrozenSet[Coord], level: Level) -> bool:
    for b in new_boxes:
        if corner_deadlock(b, level, level.goals):
            return True
    return False

# ---------- Moves (with push rules) ----------

def legal_moves(state: State, level: Level):
    """
    Yield (move_char, next_state).
    lower = walk (u/d/l/r), UPPER = push (U/D/L/R).
    """
    p_i, p_j = state.player
    for m, (di, dj) in DIRS.items():
        npos = (p_i + di, p_j + dj)
        if npos in level.walls:
            continue
        if npos in state.boxes:
            # Attempt to push
            beyond = (npos[0] + di, npos[1] + dj)
            if (beyond in level.walls) or (beyond in state.boxes):
                continue
            new_boxes = set(state.boxes)
            new_boxes.remove(npos)
            new_boxes.add(beyond)
            new_boxes_f = frozenset(new_boxes)
            if any_simple_deadlock(new_boxes_f, level):
                continue
            yield (DIR_UPPER[m], State(npos, new_boxes_f))
        else:
            yield (DIR_LOWER[m], State(npos, state.boxes))

# ---------- Heuristic (A*) ----------

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic(state: State, level: Level) -> int:
    boxes = [b for b in state.boxes if b not in level.goals]
    goals = [g for g in level.goals if g not in state.boxes]
    if not boxes:
        return 0
    goals = goals[:]
    total = 0
    for b in boxes:
        g, dist = min(((g, manhattan(b, g)) for g in goals), key=lambda x: x[1])
        total += dist
        goals.remove(g)
    pb = min((manhattan(state.player, b) for b in state.boxes), default=0)
    return total + (pb // 4)

# ---------- PQ ----------

class PQ:
    def __init__(self):
        self.h = []
        self.c = 0
    def push(self, prio, item):
        heapq.heappush(self.h, (prio, self.c, item))
        self.c += 1
    def pop(self):
        return heapq.heappop(self.h)[2]
    def __len__(self):
        return len(self.h)

# ---------- Solvers ----------

def reconstruct(moves: List[str]) -> str:
    return ''.join(moves)

def bfs(level: Level, start: State) -> Optional[str]:
    q = deque([start])
    parent = {start: (None, '')}
    visited = {start}
    while q:
        s = q.popleft()
        if is_goal_state(s, level):
            path = []
            cur = s
            while parent[cur][0] is not None:
                path.append(parent[cur][1]); cur = parent[cur][0]
            return reconstruct(path[::-1])
        for mv, ns in legal_moves(s, level):
            if ns not in visited:
                visited.add(ns); parent[ns] = (s, mv); q.append(ns)
    return None

def dfs(level: Level, start: State, depth_limit: int = 20000) -> Optional[str]:
    stack = [start]
    parent = {start: (None, '')}
    visited = {start}
    steps = 0
    while stack:
        s = stack.pop(); steps += 1
        if steps > depth_limit:
            return None
        if is_goal_state(s, level):
            path = []
            cur = s
            while parent[cur][0] is not None:
                path.append(parent[cur][1]); cur = parent[cur][0]
            return reconstruct(path[::-1])
        for mv, ns in legal_moves(s, level):
            if ns not in visited:
                visited.add(ns); parent[ns] = (s, mv); stack.append(ns)
    return None

def ucs(level: Level, start: State) -> Optional[str]:
    pq = PQ()
    pq.push(0, start)
    parent = {start: (None, '', 0)}
    best_cost = {start: 0}
    while len(pq):
        s = pq.pop()
        g = best_cost[s]
        if is_goal_state(s, level):
            path = []
            cur = s
            while parent[cur][0] is not None:
                path.append(parent[cur][1]); cur = parent[cur][0]
            return reconstruct(path[::-1])
        for mv, ns in legal_moves(s, level):
            ng = g + 1
            if ns not in best_cost or ng < best_cost[ns]:
                best_cost[ns] = ng
                parent[ns] = (s, mv, ng)
                pq.push(ng, ns)
    return None

def astar(level: Level, start: State) -> Optional[str]:
    pq = PQ()
    h0 = heuristic(start, level)
    pq.push(h0, start)
    parent = {start: (None, '', 0)}
    best_g = {start: 0}
    while len(pq):
        s = pq.pop()
        g = best_g[s]
        if is_goal_state(s, level):
            path = []
            cur = s
            while parent[cur][0] is not None:
                path.append(parent[cur][1]); cur = parent[cur][0]
            return reconstruct(path[::-1])
        for mv, ns in legal_moves(s, level):
            ng = g + 1
            if ns not in best_g or ng < best_g[ns]:
                best_g[ns] = ng
                parent[ns] = (s, mv, ng)
                pq.push(ng + heuristic(ns, level), ns)
    return None

# ---------- Streaming (live) A* for visualization ----------

def astar_yield(level: Level, start: State) -> Iterator[Tuple[State, Optional[str]]]:
    """
    A* that yields (state, move_char) as states enter the frontier/best-g map.
    Yields goal state with mv=None when goal is reached.
    """
    pq = PQ()
    h0 = heuristic(start, level)
    pq.push(h0, start)
    best_g = {start: 0}
    yield start, ''  # initial state

    while len(pq):
        s = pq.pop()
        g = best_g[s]
        if is_goal_state(s, level):
            yield s, None
            return
        for mv, ns in legal_moves(s, level):
            ng = g + 1
            if ns not in best_g or ng < best_g[ns]:
                best_g[ns] = ng
                pq.push(ng + heuristic(ns, level), ns)
                # stream this discovered better state immediately
                yield ns, mv

def solve_sokoban(level_lines: Iterable[str], method: str = "astar") -> Optional[str]:
    """
    Convenience wrapper that parses a text level and runs the chosen search.
    Returns a move string (lowercase = walk, UPPER = push) or None if unsolved.
    """
    level, start = parse_level(level_lines)
    if method == "bfs":
        return bfs(level, start)
    elif method == "dfs":
        return dfs(level, start)
    elif method == "ucs":
        return ucs(level, start)
    elif method == "astar":
        return astar(level, start)
    else:
        raise ValueError("Unknown method. Choose from: bfs, dfs, ucs, astar.")


# ---------- Demo ----------
if __name__ == "__main__":
    demo = [
        "#####",
        "#.  #",
        "# $ #",
        "#  @#",
        "#####",
    ]
    sol = astar(*parse_level(demo))
    print("Solution:", sol if sol is not None else "No solution")
