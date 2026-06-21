"""Approximate optimal path length in the continuous environment.

The discrete assignment used BFS on the grid. Here the agent moves with
primitives (forward `MOVE_DISTANCE`, turn +/- `TURN_ANGLE`), so the analogue is
a **state-lattice BFS** over those exact primitives:

    state  = (x, y, theta)
    edges  = {forward, turn left, turn right}   (each costs 1 action)
    blocked / target tests = the *same* shapely checks the environment uses.

Because every edge costs 1, BFS returns the minimum **number of agent actions**
to drive the disc's centre into a target cell -- the continuous analogue of the
discrete BFS step count, and the right denominator for SPL.

It is *approximate* because (a) the continuous position is de-duplicated on a
grid of resolution `pos_res`, and (b) headings are taken on the canonical
`TURN_ANGLE` lattice (the agent's real start heading is random, so by default
the initial orientation is treated as free -- see `free_initial_heading`).
"""

from __future__ import annotations

from collections import deque
from math import cos, sin, pi
from pathlib import Path

import numpy as np
from shapely.geometry import Point, LineString


def grid_start_cell(grid_fp) -> tuple[int, int]:
    """Returns the grid's start cell (value 4), or the first empty cell (0)."""
    cells = np.load(Path(grid_fp))
    starts = np.argwhere(cells == 4)
    if len(starts) > 0:
        return int(starts[0][0]), int(starts[0][1])
    empty = np.argwhere(cells == 0)
    if len(empty) == 0:
        raise ValueError(f"No start (4) or empty (0) cell found in {grid_fp}")
    return int(empty[0][0]), int(empty[0][1])


def approx_optimal_steps(
    env,
    start_xy: tuple[float, float],
    *,
    pos_res: float = 0.1,
    free_initial_heading: bool = True,
    start_theta: float = 0.0,
    max_expansions: int = 3_000_000,
    return_path: bool = False,
):
    """Minimum number of agent actions from `start_xy` to any target.

    Args:
        env: A reset `EnvironmentContinuous` (provides `walls`, `targets`,
            `AGENT_RADIUS`, `MOVE_DISTANCE`, `TURN_ANGLE`).
        start_xy: Continuous start position (the agent's start cell centre).
        pos_res: Grid resolution (m) used to de-duplicate visited positions.
        free_initial_heading: If True, all canonical headings are available at
            the start at cost 0 (initial orientation free). If False, only the
            heading bin nearest `start_theta` is seeded.
        start_theta: Used when `free_initial_heading` is False.
        max_expansions: Safety cap; returns None if exceeded.
        return_path: If True, also returns the list of (x, y) waypoints.

    Returns:
        steps (int) or None if unreachable; or (steps, path) if return_path.
    """
    walls = env.walls
    targets = env.targets
    radius = env.AGENT_RADIUS
    move = env.MOVE_DISTANCE
    turn = env.TURN_ANGLE

    if not targets:
        return (None, []) if return_path else None

    n_bins = max(1, round(2 * pi / turn))
    sx, sy = float(start_xy[0]), float(start_xy[1])

    def covered(x, y) -> bool:
        p = Point(x, y)
        return any(poly.covers(p) for poly in targets)

    def blocked(x, y, nx, ny) -> bool:
        # Same swept-disc test as EnvironmentContinuous._attempt_forward.
        return walls.distance(LineString([(x, y), (nx, ny)])) < radius

    def key(x, y, b):
        return (round(x / pos_res), round(y / pos_res), b % n_bins)

    if covered(sx, sy):
        return (0, [(sx, sy)]) if return_path else 0

    came_from: dict = {}     # key -> parent key (for path reconstruction)
    node_pos: dict = {}      # key -> (x, y)
    visited = set()
    q: deque = deque()

    if free_initial_heading:
        seed_bins = range(n_bins)
    else:
        seed_bins = [round(start_theta / turn) % n_bins]

    for b in seed_bins:
        theta = b * turn
        k = key(sx, sy, b)
        if k not in visited:
            visited.add(k)
            came_from[k] = None
            node_pos[k] = (sx, sy)
            q.append((sx, sy, theta, b, 0, k))

    def _reconstruct(end_key):
        path = []
        k = end_key
        while k is not None:
            path.append(node_pos[k])
            k = came_from.get(k)
        path.reverse()
        return path

    expansions = 0
    while q:
        x, y, theta, b, d, k = q.popleft()
        expansions += 1
        if expansions > max_expansions:
            return (None, []) if return_path else None

        # forward
        nx = x + move * cos(theta)
        ny = y - move * sin(theta)        # matches env: ny = y - move * sin(theta)
        if not blocked(x, y, nx, ny):
            if covered(nx, ny):
                fkey = key(nx, ny, b)
                came_from[fkey] = k
                node_pos[fkey] = (nx, ny)
                if return_path:
                    return d + 1, _reconstruct(fkey)
                return d + 1
            nk = key(nx, ny, b)
            if nk not in visited:
                visited.add(nk)
                came_from[nk] = k
                node_pos[nk] = (nx, ny)
                q.append((nx, ny, theta, b, d + 1, nk))

        # turn left / right (position unchanged)
        for db in (1, -1):
            nb = (b + db) % n_bins
            nk = key(x, y, nb)
            if nk not in visited:
                visited.add(nk)
                came_from[nk] = k
                node_pos[nk] = (x, y)
                q.append((x, y, nb * turn, nb, d + 1, nk))

    return (None, []) if return_path else None


def approx_optimal_for_grid(
    grid_fp,
    start_pos: tuple[int, int] | None = None,
    *,
    agent_radius: float = 0.2,
    move_distance: float = 0.2,
    turn_angle_deg: float = 15.0,
    pos_res: float = 0.1,
    free_initial_heading: bool = True,
    return_path: bool = False,
):
    """Convenience wrapper that builds the env and computes the optimal steps.

    If `start_pos` is None, the grid's own start cell (value 4) is used.
    """
    from math import radians
    from world.environment_continuous import EnvironmentContinuous

    if start_pos is None:
        start_pos = grid_start_cell(grid_fp)

    env = EnvironmentContinuous(
        grid_fp=Path(grid_fp),
        no_gui=True,
        sigma=0.0,
        agent_start_pos=start_pos,
        random_seed=0,
        agent_radius=agent_radius,
        move_distance=move_distance,
        turn_angle=radians(turn_angle_deg),
    )
    env.reset()
    start_xy = (env.x, env.y)
    result = approx_optimal_steps(
        env, start_xy, pos_res=pos_res,
        free_initial_heading=free_initial_heading, return_path=return_path,
    )
    return (env, result) if return_path else result


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parent
    grid_fp = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else repo_root / "grid_configs" / "restaurant_test.npy"
    # Default to the grid's own start cell (value 4) instead of a fixed corner.
    start = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 \
        else grid_start_cell(grid_fp)

    env, (steps, path) = approx_optimal_for_grid(grid_fp, start, return_path=True)
    print(f"Approx optimal from {start} on {grid_fp.name}: {steps} actions")
    if steps is not None:
        img = env.trajectory_image(path)
        out = Path("results") / f"optimal_{grid_fp.stem}_{start[0]}_{start[1]}.png"
        out.parent.mkdir(exist_ok=True)
        img.save(out)
        print(f"Saved optimal-path image to {out}")
