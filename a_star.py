#file that calculates the shortest path using A* algorithm (which discretizes the environment because of the actions finite action space)
import heapq
from math import cos, sin, pi, hypot,radians
from shapely.geometry import LineString, Point

from world.environment_continuous import EnvironmentContinuous 


def discretize(v, res=0.05):
    #discretize the model by rounding continuous values to the nearest point on the lattice. So we can find optimal route (i.e. instead of having infinite possible routes)
    return round(round(v / res) * res, 6)



def find_astar_path(env: "EnvironmentContinuous"):
    """
    Lattice A* on the (x, y, theta) state space.

    Returns a list of (x, y, theta) waypoints from start to target,
    or None if no path exists.
    
    Assumes env has already been reset() so geometry is built and
    the agent pose (env.x, env.y, env.theta) is set.
    """
    # ── constants ──────────────────────────────────────────────────────
    MOVE   = env.MOVE_DISTANCE          # 0.2 m
    DTHETA = env.TURN_ANGLE             # radians(15)
    R      = env.AGENT_RADIUS           # 0.2 m
    walls  = env.walls                  # Shapely geometry

    # 24 canonical headings, stored as index 0..23
    N_HEADINGS = round(2 * pi / DTHETA)   # 24
    
    def theta_idx(theta: float) -> int:
        """Map any angle to 0..23."""
        return round(((theta % (2 * pi)) / DTHETA)) % N_HEADINGS

    def idx_to_theta(i: int) -> float:
        return i * DTHETA

    # ── goal test ──────────────────────────────────────────────────────
    def at_goal(x, y) -> bool:
        disc = Point(x, y).buffer(env.AGENT_RADIUS)
        return any(poly.intersects(disc) for poly in env.targets)

    # ── heuristic: straight-line dist to nearest target centre ─────────
    target_centres = [
        ((poly.bounds[0] + poly.bounds[2]) / 2,
         (poly.bounds[1] + poly.bounds[3]) / 2)
        for poly in env.targets
    ]

    def heuristic(x, y) -> float:
        return min(hypot(x - cx, y - cy) for cx, cy in target_centres)

    # ── collision check for a forward move ─────────────────────────────
    def move_is_free(x, y, theta_i) -> tuple[float, float] | None:
        """
        Returns (nx, ny) if the forward step is collision-free, else None.
        """
        th = idx_to_theta(theta_i)
        nx = discretize(x + MOVE * cos(th), res=0.05)
        ny = discretize(y - MOVE * sin(th), res=0.05)
        motion = LineString([(x, y), (nx, ny)])
        if walls.distance(motion) < R:
            return None
        return nx, ny

    # ── start node ─────────────────────────────────────────────────────
    sx = discretize(env.x, res=0.05)
    sy = discretize(env.y, res=0.05)
    st = theta_idx(env.theta)
    start = (sx, sy, st)

    # ── A* ─────────────────────────────────────────────────────────────
    # heap entries: (f, g, node)
    # node = (x, y, theta_index)
    ACTION_COST = 1.0    # cost of actions

    g_score = {start: 0.0}
    came_from = {start: None}
    h0 = heuristic(sx, sy)
    heap = [(h0, 0.0, start)]

    while heap:
        f, g, node = heapq.heappop(heap)
        x, y, ti = node

        # Skip stale heap entries
        if g > g_score.get(node, float("inf")):
            continue

        if at_goal(x, y):
            # ── reconstruct path ───────────────────────────────────────
            path = []
            cur = node
            while cur is not None:
                cx, cy, cti = cur
                path.append((cx, cy, idx_to_theta(cti)))
                cur = came_from[cur]
            path.reverse()
            return path

        # ── expand neighbours ──────────────────────────────────────────
        # Action 0: move forward
        result = move_is_free(x, y, ti)
        if result is not None:
            nx, ny = result
            neighbour = (nx, ny, ti)
            ng = g + ACTION_COST
            if ng < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = ng
                came_from[neighbour] = node
                heapq.heappush(heap, (ng + heuristic(nx, ny), ng, neighbour))

        # Action 1: turn left  (+DTHETA)
        neighbour = (x, y, (ti + 1) % N_HEADINGS)
        ng = g + ACTION_COST
        if ng < g_score.get(neighbour, float("inf")):
            g_score[neighbour] = ng
            came_from[neighbour] = node
            heapq.heappush(heap, (ng + heuristic(x, y), ng, neighbour))

        # Action 2: turn right (−DTHETA)
        neighbour = (x, y, (ti - 1) % N_HEADINGS)
        ng = g + ACTION_COST
        if ng < g_score.get(neighbour, float("inf")):
            g_score[neighbour] = ng
            came_from[neighbour] = node
            heapq.heappush(heap, (ng + heuristic(x, y), ng, neighbour))

    return None  # no path found



def path_to_actions(path: list[tuple[float, float, float]]) -> list[int]:
    """Convert a waypoint path back to the action sequence [0, 1, 2]."""
    DTHETA = radians(15)
    N_HEADINGS = 24

    def theta_idx(theta):
        return round(((theta % (2 * pi)) / DTHETA)) % N_HEADINGS

    actions = []
    for i in range(len(path) - 1):
        _, _, t0 = path[i]
        x1, y1, t1 = path[i + 1]
        x0, y0, _  = path[i]

        ti0 = theta_idx(t0)
        ti1 = theta_idx(t1)

        if ti0 == ti1:
            actions.append(0)   # moved forward
        elif (ti0 + 1) % N_HEADINGS == ti1:
            actions.append(1)   # turned left
        else:
            actions.append(2)   # turned right
    return actions

def path_to_xy(path: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Strips theta, keeping only (x, y) — what trajectory_image() expects."""
    return [(x, y) for x, y, _ in path]