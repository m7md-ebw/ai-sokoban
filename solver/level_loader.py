import io

def load_boxoban_levels(path: str):
    """Reads Sokoban levels from a text file.
    - Lines starting with ';' are comments/level numbers.
    - Blank lines separate levels.
    """
    levels, cur = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if line.startswith(';'):
                continue
            if not line.strip():
                if cur:
                    levels.append(cur)
                    cur = []
                continue
            cur.append(line)
    if cur:
        levels.append(cur)
    return levels
