# run_planner.py  (new small script, or add to planner.py's __main__ block)
from pathlib import Path
from datetime import datetime

from world.environment_continuous import EnvironmentContinuous
from world.helpers import save_results
from a_star import find_astar_path, path_to_xy, path_to_actions


def plan_and_save(grid_fp: Path, agent_start_pos=None, show_images=False):
    env = EnvironmentContinuous(grid_fp=grid_fp,
                                no_gui=True,
                                agent_start_pos=agent_start_pos,
                                target_fps=-1,
                                move_distance=0.5,
                                agent_radius=0.2)
    env.reset()

    path = find_astar_path(env)
    if path is None:
        print("No path found.")
        return

    xy_path = path_to_xy(path)
    print(xy_path)
    img = env.trajectory_image(xy_path)

    file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + "_astar"
    save_results(file_name, env.world_stats, img, show_images)
    print(f"Saved trajectory image for {len(path)}-step A* path.")


if __name__ == "__main__":
    import sys
    repo_root = Path(__file__).resolve().parent   # not .parents[1]
    grid_fp = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else repo_root / "grid_configs" / "restaurant_medium.npy"
    plan_and_save(grid_fp, agent_start_pos=(18,10))