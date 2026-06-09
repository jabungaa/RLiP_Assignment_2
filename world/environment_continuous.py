"""
Continuous Environment.

Run this file to test the environment with a random agent.
"""
import io
import random
from math import cos, sin, pi, degrees, radians
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime

import numpy as np
from tqdm import trange
from PIL import Image
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union

from agents import BaseAgent
from world.helpers import save_results
from world.grid_continuous import GridContinuous


class EnvironmentContinuous:
    # Agent / motion parameters (metres, radians). One grid cell = one metre.
    AGENT_RADIUS = 0.2
    MOVE_DISTANCE = 0.2
    TURN_ANGLE = radians(15)

    N_ACTIONS = 3        # 0 = forward, 1 = turn left, 2 = turn right
    STATE_SIZE = 3       # (x, y, theta)

    def __init__(self,
                 grid_fp: Path,
                 no_gui: bool = False,
                 sigma: float = 0.,
                 agent_start_pos: tuple[int, int] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0):
        """Creates the continuous Grid Environment for the Reinforcement Learning robot
        from the provided file.

        This environment follows the general principles of reinforcment
        learning. It can be thought of as a function E : action -> observation
        where E is the environment represented as a function.
        
        Args:
            grid_fp: Path to the grid file to use.
            no_gui: True if no GUI is desired.
            sigma: The stochasticity of the environment. The probability that
                the agent makes the move that it has provided as an action is
                calculated as 1-sigma.
            agent_start_pos: Tuple where each agent should start.
                If None is provided, then a random start position is used.
            reward_fn: Custom reward function to use. 
            target_fps: How fast the simulation should run if it is being shown
                in a GUI. If in no_gui mode, then the simulation will run as fast as
                possible. We may set a low FPS so we can actually see what's
                happening. Set to 0 or less to unlock FPS.
            random_seed: The random seed to use for this environment. If None
                is provided, then the seed will be set to 0.
        """
        random.seed(random_seed)

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        self.grid_fp = grid_fp

        # Initialize other variables
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma

        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI specific code: Set up the environment as a blank state.
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

        # Lazily created matplotlib handles for rendering.
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reset_info() -> dict:
        """Resets the info dictionary.

        info is a dict with information of the most recent step
        consisting of whether the target was reached or the agent
        moved and the updated agent position.
        """
        return {"target_reached": False,
                "agent_moved": False,
                "collided": False,
                "actual_action": None,
                "pose": None}

    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary.

        world_stats is a dict with information about the 
        environment since last env.reset(). Basically, it
        accumulates information.
        """
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_turns": 0,
                "total_targets_reached": 0}

    # ------------------------------------------------------------------ #
    # Geometry / start position
    # ------------------------------------------------------------------ #
    def _build_geometry(self):
        """Builds the Shapely wall and target geometry from the grid."""
        grid = GridContinuous.from_file(self.grid_fp)
        self.cells = grid.cells
        self.n_cols, self.n_rows = self.cells.shape

        segments = grid.to_segments()
        self.wall_segments = segments
        self.walls = unary_union([LineString(s) for s in segments])

        # Each target cell becomes a unit square region to be reached.
        self.targets = [box(int(c), int(r), int(c) + 1, int(r) + 1)
                        for c, r in np.argwhere(self.cells == 3)]
        self.n_targets_total = len(self.targets)

    def _validate_start_cell(self, cell: tuple[int, int]):
        c, r = cell
        if not (0 <= c < self.n_cols and 0 <= r < self.n_rows):
            raise ValueError(f"Start cell {cell} is out of bounds "
                             f"({self.n_cols}x{self.n_rows}).")
        if self.cells[c, r] != 0:
            names = {0: "empty", 1: "boundary", 2: "obstacle", 3: "target"}
            raise ValueError(
                f"Start cell {cell} is a {names.get(int(self.cells[c, r]))} "
                f"cell. The agent can only start on an empty cell.")

    def _initialize_agent_pose(self):
        """Sets the agent's continuous pose (centre of a free cell + heading)."""
        if self.agent_start_pos is not None:
            cell = (int(self.agent_start_pos[0]), int(self.agent_start_pos[1]))
            self._validate_start_cell(cell)
        else:
            free = np.argwhere(self.cells == 0)
            idx = random.randint(0, len(free) - 1)
            cell = (int(free[idx][0]), int(free[idx][1]))

        # Centre of the chosen cell, with a random initial heading. ?? may need to be fixed
        self.x = cell[0] + 0.5
        self.y = cell[1] + 0.5
        self.theta = random.uniform(-pi, pi)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float32)

    def reset(self, **kwargs) -> np.ndarray:
        """Reset the environment to an initial state.

        You can fit it keyword arguments which will overwrite the 
        initial arguments provided when initializing the environment.

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        Returns:
             initial state.
        """
        for k, v in kwargs.items():
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 0. if v <= 0 else 1. / v
                case "sigma" | "reward_fn" | "random_seed":
                    raise ValueError(f"{k} cannot be changed after init.")
                case _:
                    raise ValueError(f"{k} is not a valid keyword argument.")

        self._build_geometry()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        self._initialize_agent_pose()
        self.info["pose"] = self._get_state()

        if not self.no_gui:
            self.render()

        return self._get_state()

    def _attempt_forward(self) -> bool:
        """Tries to move the agent forward; returns True if it moved.

        The move is blocked (collision) if the agent would come within
        ``AGENT_RADIUS`` of any wall.
        """
        nx = self.x + self.MOVE_DISTANCE * cos(self.theta)
        ny = self.y + self.MOVE_DISTANCE * sin(self.theta)
        motion = LineString([(self.x, self.y), (nx, ny)])
        if self.walls.distance(motion) < self.AGENT_RADIUS:
            return False  # collision: stay in place
        self.x, self.y = nx, ny
        return True

    def _collect_targets(self) -> bool:
        """Removes any target the agent disc now overlaps; returns True if any."""
        center = Point(self.x, self.y)
        reached = False
        remaining = []
        for poly in self.targets:
            if poly.covers(center):
                reached = True
            else:
                remaining.append(poly)
        self.targets = remaining
        if reached:
            self.terminal_state = True
        return reached

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Advances the environment by one action.

        Actions: 0 = forward, 1 = turn left, 2 = turn right.

        Returns:
            (state, reward, terminal, info)
        """
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}; expected 0, 1 or 2.")

        self.info = self._reset_info()
        self.world_stats["total_steps"] += 1

        start_time = time()

        # Environment stochasticity.
        if random.random() <= self.sigma:
            actual_action = random.randint(0, 2)
        else:
            actual_action = action
        self.info["actual_action"] = actual_action

        collided = False
        moved = False
        target_reached = False

        if actual_action == 0:                       # forward
            moved = self._attempt_forward()
            if moved:
                self.world_stats["total_agent_moves"] += 1
                target_reached = self._collect_targets()
                if target_reached:
                    self.world_stats["total_targets_reached"] += 1
            else:
                collided = True
                self.world_stats["total_failed_moves"] += 1
        else:                                        # turn in place
            self.theta += self.TURN_ANGLE if actual_action == 1 \
                else -self.TURN_ANGLE
            # Wrap to [-pi, pi].
            self.theta = (self.theta + pi) % (2 * pi) - pi
            self.world_stats["total_turns"] += 1

        reward = self.reward_fn(collided, target_reached, moved)
        self.world_stats["cumulative_reward"] += reward

        self.info["agent_moved"] = moved
        self.info["collided"] = collided
        self.info["target_reached"] = target_reached
        self.info["pose"] = self._get_state()

        if not self.no_gui:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.render()

        return self._get_state(), reward, self.terminal_state, self.info

    def _default_reward_function(self, collided: bool, target_reached: bool,
                                 moved: bool) -> float:
        """Default reward: reach target (+10), bump wall (-5), else step (-1)."""
        if target_reached:
            return 10.0
        if collided:
            return -5.0
        return -1.0

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _draw_scene(self, ax, path: list[tuple[float, float]] | None = None):
        """Draws walls, targets, agent and (optionally) a trajectory on ``ax``."""
        from matplotlib.patches import Circle, Rectangle

        ax.clear()
        for (x1, y1), (x2, y2) in self.wall_segments:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2, zorder=1)
        # Targets (remaining + already-collected, drawn from total set).
        for poly in self.targets:
            minx, miny, maxx, maxy = poly.bounds
            ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny,
                                   facecolor="#4CAF50", alpha=0.5, zorder=0))
        if path is not None and len(path) > 1:
            px, py = zip(*path)
            ax.plot(px, py, color="#1f77b4", linewidth=1.2, alpha=0.8, zorder=2)

        # Agent disc and heading.
        ax.add_patch(Circle((self.x, self.y), self.AGENT_RADIUS,
                            facecolor="#E1602F", edgecolor="black", zorder=3))
        ax.plot([self.x, self.x + self.AGENT_RADIUS * cos(self.theta)],
                [self.y, self.y + self.AGENT_RADIUS * sin(self.theta)],
                color="black", linewidth=1.5, zorder=4)

        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    def render(self):
        """Live matplotlib render of the current state."""
        import matplotlib.pyplot as plt
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 7))
        self._draw_scene(self._ax)
        self._fig.canvas.draw_idle()
        plt.pause(max(self.target_spf, 1e-3))

    def trajectory_image(self, path: list[tuple[float, float]]) -> Image.Image:
        """Renders the wall geometry plus an agent trajectory to a PIL image."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_scene(ax, path=path)
        if path:
            ax.plot(path[0][0], path[0][1], "o", color="gold",
                    markersize=10, markeredgecolor="black", zorder=5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    @staticmethod
    def evaluate_agent(grid_fp: Path,
                       agent: BaseAgent,
                       max_steps: int,
                       sigma: float = 0.,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False):
        """Evaluates a trained agent and saves stats + a trajectory image."""
        env = EnvironmentContinuous(grid_fp=grid_fp,
                                    no_gui=True,
                                    sigma=sigma,
                                    agent_start_pos=agent_start_pos,
                                    target_fps=-1,
                                    random_seed=random_seed)
        state = env.reset()
        path = [(env.x, env.y)]

        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)
            path.append((env.x, env.y))
            if terminated:
                break

        env.world_stats["targets_remaining"] = len(env.targets)

        path_image = env.trajectory_image(path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        save_results(file_name, env.world_stats, path_image, show_images)


if __name__ == "__main__":
    import sys

    class _Random3Agent(BaseAgent):
        def take_action(self, state):
            return random.randint(0, 2)

        def update(self, state, reward, action):
            pass

    repo_root = Path(__file__).resolve().parents[1]
    grid_fp = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else repo_root / "grid_configs" / "A1_grid.npy"
    EnvironmentContinuous.evaluate_agent(grid_fp, _Random3Agent(),
                                         max_steps=500, random_seed=0)
