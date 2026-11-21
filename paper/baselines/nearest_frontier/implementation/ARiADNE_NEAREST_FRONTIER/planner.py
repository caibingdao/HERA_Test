"""Nearest reachable frontier on a four-connected occupancy grid."""
from collections import deque

import numpy as np


def nearest_frontier_path(belief, start, *, free=255, unknown=127):
    """Return an inclusive shortest (row, column) path to a frontier beyond start.

    Frontier cells are observed free cells with a four-connected unknown
    neighbor. Equal distances use a fixed up, left, right, down search order.
    Return an empty list when there is no reachable target to move toward.
    """
    belief = np.asarray(belief)
    if belief.ndim != 2:
        raise ValueError("belief must be a two-dimensional grid")
    height, width = belief.shape
    row, col = start
    if not (0 <= row < height and 0 <= col < width) or belief[row, col] != free:
        raise ValueError("start must be an observed free cell")

    unknown_cells = belief == unknown
    frontier = np.zeros(belief.shape, dtype=bool)
    frontier[1:, :] |= unknown_cells[:-1, :]
    frontier[:-1, :] |= unknown_cells[1:, :]
    frontier[:, 1:] |= unknown_cells[:, :-1]
    frontier[:, :-1] |= unknown_cells[:, 1:]
    traversable = (belief == free).ravel()
    targets = frontier.ravel() & traversable
    origin = row * width + col
    targets[origin] = False
    if not targets.any():
        return []

    parent = np.full(belief.size, -1, dtype=np.intp)
    parent[origin] = origin
    queue = deque([origin])
    while queue:
        cell = queue.popleft()
        if targets[cell]:
            path = [divmod(cell, width)]
            while cell != origin:
                cell = int(parent[cell])
                path.append(divmod(cell, width))
            return path[::-1]

        row, col = divmod(cell, width)
        for next_row, next_col in ((row - 1, col), (row, col - 1),
                                   (row, col + 1), (row + 1, col)):
            if 0 <= next_row < height and 0 <= next_col < width:
                neighbor = next_row * width + next_col
                if traversable[neighbor] and parent[neighbor] == -1:
                    parent[neighbor] = cell
                    queue.append(neighbor)
    return []


def next_waypoint(path, max_cells):
    """Take the first straight segment, capped at max_cells grid edges."""
    if not path:
        return None
    start_row, start_col = path[0]
    first_row, first_col = path[1]
    delta_row, delta_col = first_row - start_row, first_col - start_col
    waypoint = path[1]
    for distance, cell in enumerate(path[2:max_cells + 1], start=2):
        if cell != (start_row + distance * delta_row, start_col + distance * delta_col):
            break
        waypoint = cell
    return waypoint
