"""Standalone PPO success-rate CLI runner.

The training/evaluation logic now lives in `train_ppo.py` (so it can be reused
by `bayesian_search.py` and `train_and_evaluate.py`); this file is just the
command-line front-end around it.

Example:
    python test_ppo.py --grid grid_configs/A1_grid.npy --episodes 1000
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from math import radians
from pathlib import Path
import copy

from matplotlib.pyplot import step
import numpy as np

from train_ppo import (
    REWARD_SCALES,
    parse_start_pos,
    parse_hidden_sizes,
    make_training_start_sampler,
    train_ppo,
    evaluate_ppo,
)

RESULTS_DIR = Path("results")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate PPO_agent only.")
    parser.add_argument("--grid", type=Path, default=Path("grid_configs/A1_grid.npy"))
    parser.add_argument("--start_pos", type=str, default=None,
                        help="Start cell as col,row. If omitted, first start/empty cell is used.")
    parser.add_argument("--reward", choices=("default", "high"), default="high")
    parser.add_argument("--sigma", "--alpha", dest="sigma", type=float, default=0.0,
                        help="Environment stochasticity (also accepted as --alpha).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu",
                        help='Torch device to use: "cpu", "cuda", or "cuda:0".')

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--train_start_mode", choices=("random", "fixed"), default="random",
                        help="Training start positions. Evaluation always uses --start_pos.")
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_max_steps", type=int, default=1000)

    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=64,
                        help="Transitions per gradient step within each epoch. Default 64.")
    parser.add_argument("--replay_capacity", type=int, default=0,
                        help="Off-policy replay buffer size. 0 = on-policy PPO (default).")
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--hidden_sizes", type=str, default="128,128",
                        help="Comma-separated actor/critic hidden sizes, e.g. 64,64.")
    parser.add_argument("--activation", choices=("tanh", "relu", "elu", "gelu"), default="tanh",
                        help="Hidden-layer activation function.")
    parser.add_argument("--fourier_freqs", type=int, default=0,
                        help="Fourier frequency bands for raw continuous x,y. 0 = disabled.")
    parser.add_argument("--reward_scale", type=float, default=None,
                        help="Divide all training rewards by this before the agent sees them. "
                             "Default: auto from --reward.")
    parser.add_argument("--max_grad_norm", type=float, default=1000000.0)
    parser.add_argument("--save_train_images", action="store_true",
                        help="Save a trajectory image per training episode.")
    parser.add_argument("--greedy_eval_interval", type=int, default=20,
                        help="Evaluate every N training episodes (100 runs when sigma is "
                             "non-zero, otherwise one greedy run). 0 disables checks.")

    parser.add_argument("--agent_radius", type=float, default=0.2,
                        help="Agent disc radius in metres. Default 0.2.")
    parser.add_argument("--move_distance", type=float, default=0.2,
                        help="Distance moved per forward action in metres. Default 0.2.")
    parser.add_argument("--turn_angle_deg", type=float, default=15.0,
                        help="Angle rotated per turn action in degrees. Default 15.")

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    try:
        from agents.PPO import PPO_agent
        import torch
        from world.environment_continuous import EnvironmentContinuous
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for PPO_agent. Install project dependencies "
                "with `pip install -r requirements.txt`, then rerun this script."
            ) from exc
        if exc.name == "pygame":
            raise SystemExit(
                "pygame is required by world.environment_continuous. Install project "
                "dependencies with `pip install -r requirements.txt`, then rerun."
            ) from exc
        raise

    Environment = EnvironmentContinuous
    reward_fn = {
        "default": Environment._default_reward_function,
        "high": Environment._high_reward_function,
    }[args.reward]
    if args.reward_scale is None:
        args.reward_scale = REWARD_SCALES[args.reward]

    start_pos = parse_start_pos(args.start_pos, args.grid)
    start_sampler = make_training_start_sampler(
        args.grid, fixed_start=start_pos, mode=args.train_start_mode, seed=args.seed,
    )
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    cuda_available = torch.cuda.is_available()
    if args.device.startswith("cuda") and not cuda_available:
        raise SystemExit(
            f"Requested --device {args.device}, but torch.cuda.is_available() is False."
        )

    print(f"Grid: {args.grid}")
    print(f"Start position: {start_pos}")
    print(f"Reward: {args.reward}, sigma={args.sigma}, gamma={args.gamma}")
    print(f"Reward scale: {args.reward_scale:g} (rewards divided by this)")
    print(f"Training: episodes={args.episodes}, max_steps_per_episode={args.iters}")
    print(f"Torch device requested: {args.device} (cuda available: {cuda_available})")

    env = Environment(
        grid_fp=args.grid,
        no_gui=True,
        sigma=args.sigma,
        agent_start_pos=start_pos,
        target_fps=-1,
        random_seed=args.seed,
        reward_fn=reward_fn,
        agent_radius=args.agent_radius,
        move_distance=args.move_distance,
        turn_angle=radians(args.turn_angle_deg),
    )

    from optimal_path import approx_optimal_steps
    env.reset(agent_start_pos=start_pos)
    baseline_steps = approx_optimal_steps(
        env, (env.x, env.y), free_initial_heading=True,
    )
    if baseline_steps is None:
        raise SystemExit("The heuristic planner could not find a path to the target.")
    print(f"Heuristic baseline: {baseline_steps} steps")

    agent = PPO_agent(
        grid=args.grid,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        replay_capacity=args.replay_capacity,
        rollout_steps=args.rollout_steps,
        hidden_sizes=hidden_sizes,
        reward_scale=args.reward_scale,
        max_grad_norm=args.max_grad_norm,
        activation=args.activation,
        fourier_freqs=args.fourier_freqs,
        state_size=Environment.STATE_SIZE,
        seed=args.seed,
        device=args.device,
    )
    actual_device = str(agent.device)
    print(f"Agent actual device: {actual_device}")

    run_ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    train_images_dir = None
    if args.save_train_images:
        train_images_dir = Path(__file__).parent / "results" / f"{run_ts}_training"
        train_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training images will be saved to {train_images_dir}/")

    t_train_start = time.time()

    checkpoint_eval_episodes = 100 if args.sigma != 0 else 1

    def checkpoint_evaluation():
        metrics = evaluate_ppo(
            agent=agent,
            Environment=Environment,
            grid_fp=args.grid,
            reward_fn=reward_fn,
            start_pos=start_pos,
            sigma=args.sigma,
            seed=args.seed,
            episodes=checkpoint_eval_episodes,
            max_steps=args.eval_max_steps,
            agent_radius=args.agent_radius,
            move_distance=args.move_distance,
            turn_angle_deg=args.turn_angle_deg,
            baseline_steps=baseline_steps,
            save_image=False,
        )
        if args.sigma != 0:
            stop = metrics["eval_within_150pct_baseline_rate"] == 1.0
            print(
                f"\n[Evaluation] avg steps={metrics['eval_avg_steps']:.2f}, "
                f"within 150% baseline="
                f"{metrics['eval_within_150pct_baseline_count']}/100"
            )
        else:
            stop = metrics["eval_total_successes"] == 1
            print(
                f"\n[Greedy evaluation] success={bool(stop)}, "
                f"steps={metrics['eval_steps']}"
            )
        # Keep checkpoint history compact: these are the requested summary
        # values; per-step paths remain available only for the final evaluation.
        return {
            "eval_total_successes": metrics["eval_total_successes"],
            "eval_total_episodes": metrics["eval_total_episodes"],
            "eval_success_rate": metrics["eval_success_rate"],
            "eval_avg_steps": metrics["eval_avg_steps"],
            "eval_baseline_steps": metrics["eval_baseline_steps"],
            "eval_within_150pct_baseline_count":
                metrics["eval_within_150pct_baseline_count"],
            "eval_within_150pct_baseline_rate":
                metrics["eval_within_150pct_baseline_rate"],
            "eval_spl": metrics["eval_spl"],
            "stop_training": stop,
        }

    train_metrics = train_ppo(
        agent, env, args.episodes, args.iters, start_sampler,
        train_images_dir=train_images_dir,
        greedy_eval_interval=args.greedy_eval_interval,
        greedy_eval_fn=checkpoint_evaluation,
    )
    train_metrics["train_time_s"] = round(time.time() - t_train_start, 2)

    final_eval_episodes = 100 if args.sigma != 0 else args.eval_episodes
    eval_metrics = evaluate_ppo(
        agent=agent,
        Environment=Environment,
        grid_fp=args.grid,
        reward_fn=reward_fn,
        start_pos=start_pos,
        sigma=args.sigma,
        seed=args.seed,
        episodes=final_eval_episodes,
        max_steps=args.eval_max_steps,
        agent_radius=args.agent_radius,
        move_distance=args.move_distance,
        turn_angle_deg=args.turn_angle_deg,
        baseline_steps=baseline_steps,
    )

    print("\nPPO result")
    print(f"Training successes: {train_metrics['train_total_successes']}/"
          f"{train_metrics['train_total_episodes']}")
    print(f"Training success rate: {train_metrics['train_success_rate']:.2f}")
    print(f"Eval successes: {eval_metrics['eval_total_successes']}/"
          f"{eval_metrics['eval_total_episodes']}")
    print(f"Eval success rate: {eval_metrics['eval_success_rate']:.2f}")
    print(f"Heuristic baseline steps: {baseline_steps}")
    print(f"Eval SPL: {eval_metrics['eval_spl']}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    checkpoint_path = results_dir / f"{run_ts}_checkpoint.pt"
    torch.save({
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    json_path = results_dir / f"{run_ts}_metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "checkpoint": str(checkpoint_path),
            "heuristic_baseline_steps": baseline_steps,
            **train_metrics,
            **eval_metrics,
        }, f, indent=2)
    print(f"Metrics saved to {json_path}")


if __name__ == "__main__":
    main()
