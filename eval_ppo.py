"""Evaluate a saved PPO checkpoint.

Usage:
    python eval_ppo.py                          # load latest checkpoint in results/
    python eval_ppo.py --checkpoint results/2026-06-18__12-00-00_checkpoint.pt
    python eval_ppo.py --episodes 5 --max_steps 2000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from math import radians
from pathlib import Path

import torch


RESULTS_DIR = Path(__file__).parent / "results"


def find_latest_checkpoint() -> Path:
    checkpoints = sorted(RESULTS_DIR.glob("*_checkpoint.pt"),
                         key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {RESULTS_DIR}/")
    return checkpoints[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a *_checkpoint.pt file. "
                             "Defaults to the most recently saved one in results/.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of eval episodes (overrides checkpoint setting).")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (overrides checkpoint setting).")
    parser.add_argument("--start_pos", type=str, default=None,
                        help="Override start position as col,row.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]

    from agents.PPO import PPO_agent
    from world.environment_continuous import EnvironmentContinuous

    reward_fn = {
        "default": EnvironmentContinuous._default_reward_function,
        "high":    EnvironmentContinuous._high_reward_function,
    }[saved_args["reward"]]

    hidden_sizes = tuple(
        int(x) for x in str(saved_args["hidden_sizes"]).split(",") if x.strip()
    )

    agent = PPO_agent(
        grid=Path(saved_args["grid"]),
        hidden_sizes=hidden_sizes,
        state_size=EnvironmentContinuous.STATE_SIZE,
        activation=saved_args["activation"],
        fourier_freqs=int(saved_args["fourier_freqs"]),
        reward_scale=float(saved_args["reward_scale"]),
        seed=int(saved_args["seed"]),
        device="cpu",
    )
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.set_training(False)

    if args.start_pos is not None:
        col, row = (int(v) for v in args.start_pos.split(","))
        start_pos = (col, row)
    else:
        import numpy as np
        grid = np.load(saved_args["grid"])
        starts = np.argwhere(grid == 4)
        start_pos = (int(starts[0][0]), int(starts[0][1])) if len(starts) > 0 else (int(np.argwhere(grid == 0)[0][0]), int(np.argwhere(grid == 0)[0][1]))

    episodes  = args.episodes  or int(saved_args.get("eval_episodes",  1))
    max_steps = args.max_steps or int(saved_args.get("eval_max_steps", 1000))
    seed      = args.seed      or int(saved_args.get("seed", 0))

    agent_radius    = float(saved_args.get("agent_radius",    0.2))
    move_distance   = float(saved_args.get("move_distance",   0.2))
    turn_angle_deg  = float(saved_args.get("turn_angle_deg",  45.0))
    sigma           = float(saved_args.get("sigma",           0.0))

    print(f"Grid:       {saved_args['grid']}")
    print(f"Start pos:  {start_pos}")
    print(f"Episodes:   {episodes},  max_steps: {max_steps}")

    successes = []
    last_env = last_path = None

    for ep in range(episodes):
        env = EnvironmentContinuous(
            grid_fp=Path(saved_args["grid"]),
            no_gui=True,
            sigma=sigma,
            agent_start_pos=start_pos,
            random_seed=seed + ep,
            reward_fn=reward_fn,
            target_fps=-1,
            agent_radius=agent_radius,
            move_distance=move_distance,
            turn_angle=radians(turn_angle_deg),
        )
        state = env.reset()
        path = [(env.x, env.y)]
        actions = []
        reached = False

        for step in range(max_steps):
            action = agent.take_action(state)
            actions.append(action)
            state, _reward, terminated, _info = env.step(action)
            path.append((env.x, env.y))
            if terminated:
                reached = True
                break

        successes.append(1 if reached else 0)
        last_env, last_path, last_actions = env, path, actions
        status = "SUCCESS" if reached else f"FAILED  (steps={len(path)-1})"
        print(f"  ep {ep+1}/{episodes}: {status}")

    RESULTS_DIR.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

    if last_env is not None:
        from world.helpers import save_results
        img = last_env.trajectory_image(last_path)
        save_results(f"{run_ts}_eval", last_env.world_stats, img, show_images=False)
        print(f"Trajectory image saved to results/{run_ts}_eval.png")

    total = sum(successes)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "start_pos": list(start_pos),
        "eval_total_successes": total,
        "eval_total_episodes": episodes,
        "eval_success_rate": total / episodes if episodes > 0 else 0.0,
        "eval_path": [
            [round(float(x), 4), round(float(y), 4), a]
            for (x, y), a in zip(last_path, last_actions + [None])
        ],
        "eval_steps": len(last_path) - 1,
        "eval_final_pos": [round(float(last_env.x), 4), round(float(last_env.y), 4)],
    }

    json_path = RESULTS_DIR / f"{run_ts}_eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {json_path}")
    print(f"\nEval success rate: {total}/{episodes} = {metrics['eval_success_rate']:.2f}")


if __name__ == "__main__":
    main()
