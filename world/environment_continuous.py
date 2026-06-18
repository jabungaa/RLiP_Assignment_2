"""
Continuous Environment.

Run this file to test the environment with a random agent.
"""
from pathlib import Path
import sys
# Adds the parent directory above "world" folder to Python's search path, because I was getting an error that 
# it wasn't finding the "agents" folder, when I was trying to quick-run in VS Code instead of doing it from the terminal
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

import io
import random
from math import cos, sin, pi, degrees, radians
from warnings import warn
from time import time, sleep
from datetime import datetime

import numpy as np
from tqdm import trange
from PIL import Image
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from agents.base_agent import BaseAgent
from agents.DQN_agent import DQNAgent
from world.helpers import save_results
from world.grid_continuous import GridContinuous


class EnvironmentContinuous:
    # Agent / motion parameters (metres, radians). One grid cell = one metre.
    AGENT_RADIUS = 0.2
    MOVE_DISTANCE = 0.2
    TURN_ANGLE = radians(15)

    N_ACTIONS = 4        # 0 = forward, 1 = turn left, 2 = turn right, 3= move backwards
    N_RAYS = 18 # number of rays in the lidar state representation (18 rays equal a ray every 20 degrees)
    STATE_SIZE = 4 + N_RAYS # (x, y, theta) + lidar rays

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
        random.seed(random_seed) # maybe we would like to keep this fixed instead for reproducibility in the future?

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        self.grid_fp = grid_fp

        # Initialize other variables
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
        self.MAX_LIDAR_RANGE = 5

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
        """Builds the Shapely wall and target geometry from the grid.
        Also, builds a STR tree of the obstacles to speed up ray intersection queries."""
        
        grid = GridContinuous.from_file(self.grid_fp)
        self.cells = grid.cells
        for c, r in np.argwhere(self.cells == 4):
            self.cells[c, r] = 0  # treat boundary cells as obstacles
        self.n_cols, self.n_rows = self.cells.shape

        segments = grid.to_segments()
        self.wall_segments = segments
        self.walls = unary_union([LineString(s) for s in segments])

        # Each target cell becomes a unit square region to be reached.
        self.targets = [box(int(c), int(r), int(c) + 1, int(r) + 1)
                        for c, r in np.argwhere(self.cells == 3)]
        self.n_targets_total = len(self.targets)


        # Build a tree of the obstacles to speed up ray intersection queries
        self.obstacles = list(self.walls.geoms) if hasattr(self.walls, 'geoms') else list(self.walls) # because MultiLineString is not iterable itself, 
                                                                                  # we need to extract its constituent parts
        self.wall_tree = STRtree(self.obstacles) # this is done to speed up intersection checks, 
                                           # only looking at points where the rays shoot out toward
                                           # offers signficant speed up compared to checking all obstacles (~4x on my pc)

    def _validate_start_cell(self, cell: tuple[int, int]):
        c, r = cell
        # print(f"Validating start cell {cell}...")
        if not (0 <= c < self.n_cols and 0 <= r < self.n_rows):
            raise ValueError(f"Start cell {cell} is out of bounds "
                             f"({self.n_cols}x{self.n_rows}).")
        # print(self.cells)
        # print(f"Cell value: {self.cells[c, r]}")
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

    def _get_ray(self, x, y, ray_angle, walls, max_range=5.0):
        end_x = x + max_range * cos(ray_angle)
        end_y = y - max_range * sin(ray_angle) #updated sin to negative because the y-axis is inverted in the GUI (y increases downwards)
        ray = LineString([(x, y), (end_x, end_y)])

        min_dist = max_range

        obstacles_in_the_way = self.wall_tree.query(ray) # indexes the obstacles that are in the path of the ray

        for obstacle_idx in obstacles_in_the_way:
            intersection = ray.intersection(self.obstacles[obstacle_idx])
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    dist = Point(x, y).distance(intersection)
                    min_dist = min(min_dist, dist)
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection:
                        dist = Point(x, y).distance(point)
                        min_dist = min(dist, min_dist)
        return min_dist

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        # the entire 360 degree field is covered by 2*pi radians
        ray_angles = [self.theta + i * ((2*pi)/self.N_RAYS) for i in range(self.N_RAYS)]
        lidar_rays = [self._get_ray(self.x, self.y, ray_angle, self.walls, max_range=self.MAX_LIDAR_RANGE) for ray_angle in ray_angles]
        return np.array([self.x/self.n_cols, self.y/self.n_rows, np.cos(self.theta), np.sin(self.theta), *[d/self.MAX_LIDAR_RANGE for d in lidar_rays]], dtype=np.float32) #added state normalization to [0, 1] for x, y, and lidar rays, and to [-1, 1] for cos(theta) and sin(theta) to help with training

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
        ny = self.y - self.MOVE_DISTANCE * sin(self.theta)
        motion = LineString([(self.x, self.y), (nx, ny)])
        if self.walls.distance(motion) < self.AGENT_RADIUS:
            return False  # collision: stay in place
        self.x, self.y = nx, ny
        return True

    def _attempt_backward(self) -> bool:
        """Tries to move the agent backward; returns True if it moved.

        The move is blocked (collision) if the agent would come within
        ``AGENT_RADIUS`` of any wall.
        """
        nx = self.x - self.MOVE_DISTANCE * cos(self.theta)
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

        Actions: 0 = forward, 1 = turn left, 2 = turn right, 3 = backward.

        Returns:
            (state, reward, terminal, info)
        """
        if action not in (0, 1, 2, 3):
            raise ValueError(f"Invalid action {action}; expected 0, 1, 2 or 3.")

        self.info = self._reset_info()
        self.world_stats["total_steps"] += 1

        start_time = time()

        # Environment stochasticity.
        if random.random() <= self.sigma:
            actual_action = random.randint(0, 3)
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
        if actual_action == 3:                       # backward
            moved = self._attempt_backward()
            if moved:
                self.world_stats["total_agent_moves"] += 1
                target_reached = self._collect_targets()
                if target_reached:
                    self.world_stats["total_targets_reached"] += 1
            else:
                collided = True
                self.world_stats["total_failed_moves"] += 1
        else:                                        # turn in place
            if actual_action == 1: # turn left = counter-clockwise on screen = theta increases
                self.theta += self.TURN_ANGLE
            else: # turn right = clockwise on screen = theta decreases
                self.theta -= self.TURN_ANGLE
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
    
    
    @staticmethod
    def _default_reward_function(collided: bool, target_reached: bool,
                                 moved: bool) -> float:
        """Default reward: reach target (+1), bump wall (-0.25), else step (-0.001)."""
        if target_reached:
            return 1
        if collided:
            return -0.25
        if moved:
            return -0.001
        else:
            return -0.01


    @staticmethod
    def _high_reward_function(collided: bool, target_reached: bool,
                                 moved: bool) -> float:
        """High reward function: reach target (+10000), bump wall (-5), else step (-1)."""
        if target_reached:
            return 10000.0
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
                [self.y, self.y - self.AGENT_RADIUS * sin(self.theta)],
                color="black", linewidth=1.5, zorder=4)
        
        # add lidar rays to visualizations
        ray_angles = [self.theta + i * ((2*pi)/self.N_RAYS) for i in range(self.N_RAYS)]
        ray_lengths = self._get_state()[3:] # the first 3 elements of the state are (x, y, theta), the rest are lidar rays
        for ray_angle, ray_length in zip(ray_angles, ray_lengths):
            end_x = self.x + ray_length * cos(ray_angle)
            end_y = self.y - ray_length * sin(ray_angle) #updated sin to negative because the y-axis is inverted in the GUI (y increases downwards)
            ax.plot([self.x, end_x], [self.y, end_y], color="red", linewidth=1.0, zorder=1)


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
                       show_images: bool = False,
                       no_gui = True):
        """Evaluates a trained agent and saves stats + a trajectory image."""
        env = EnvironmentContinuous(grid_fp=grid_fp,
                                    no_gui=no_gui,
                                    sigma=sigma,
                                    agent_start_pos=agent_start_pos,
                                    target_fps=100,
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
        
        # Dynamically determine prefix based on the agent's class name
        agent_class_name = type(agent).__name__.lower()
        if "ppo" in agent_class_name:
            prefix = "PPO"
        elif "dqn" in agent_class_name:
            prefix = "DQN"
        else:
            prefix = "Agent"

        # Prepend the prefix to the timestamp string
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        file_name = f"{prefix}_{timestamp}"
        
        save_results(file_name, env.world_stats, path_image, show_images)

        return env.world_stats, path

if __name__ == "__main__":
    import sys

    # class _Random3Agent(BaseAgent):
    #     def take_action(self, state):
    #         return random.randint(0, 2)

    #     def update(self, state, reward, action):
    #         pass

    agent = DQNAgent(input_dim=EnvironmentContinuous.STATE_SIZE, output_dim=EnvironmentContinuous.N_ACTIONS, \
                    gamma=0.99, learning_rate=1e-3, epsilon_start=1.0, epsilon_end=0.01, \
                    batch_size=64, replay_buffer_size=10000, target_update_frequency=1000)


    repo_root = Path(__file__).resolve().parents[1]
    grid_fp = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else repo_root / "grid_configs" / "A1_grid.npy"
    EnvironmentContinuous.evaluate_agent(grid_fp, agent, sigma=0.2,
                                         max_steps=1000, random_seed=0)
