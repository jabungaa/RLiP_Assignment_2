"""Standalone PPO success-rate runner.

This script trains and evaluates only `agents.PPO.PPO_agent`. It is intentionally
separate from `new_test.py` and `test_agents.py` so PPO experiments can be run
without also training PI, SARSA, or Monte Carlo agents.

Example:
    python test_ppo.py --grid grid_configs/small_grid.npy --episodes 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

# Goal-reward magnitude of each reward function; used to auto-set --reward_scale
# so the agent trains on rewards of magnitude ~1.
REWARD_SCALES = {
    "default": 100.0,        # goal=100, step=-1, wall=-5
    "low": 10.0,             # goal=100, step=-4, wall=-5
    "zero": 100.0,           # goal=100, step=0
    "bfs": None,             # auto-set from max_dist at runtime
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate PPO_agent only.")
    parser.add_argument("--grid", type=Path, default=Path("grid_configs/small_grid.npy"))
    parser.add_argument("--start_pos", type=str, default=None,
                        help="Start position as row,col. If omitted, first empty cell is used.")
    parser.add_argument("--reward", choices=("default", "zero", "low", "bfs"), default="low")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        help='Torch device to use: "cpu", "cuda", or "cuda:0".')

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--train_start_mode", choices=("random", "fixed"), default="fixed",
                        help="Training start positions. Evaluation always uses --start_pos.")
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_max_steps", type=int, default=500)

    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--policy_lr", type=float, default=3e-3)
    parser.add_argument("--value_lr", type=float, default=3e-3)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--hidden_sizes", type=str, default="64,64",
                        help="Comma-separated actor/critic hidden sizes, e.g. 64,64.")
    parser.add_argument("--activation", choices=("tanh", "relu", "elu", "gelu"), default="tanh",
                        help="Hidden-layer activation function.")
    parser.add_argument("--fourier_freqs", type=int, default=0,
                        help="Fourier positional encoding frequency bands. "
                             "0 = disabled (raw normalised coords). "
                             "4–6 recommended when enabled.")
    parser.add_argument("--fourier_raw", action="store_true",
                        help="Use raw integer coords for Fourier encoding instead of "
                             "normalised [0,1] coords. Base frequency π so adjacent "
                             "cells differ by a quarter-period.")
    parser.add_argument("--reward_scale", type=float, default=None,
                        help="Divide all training rewards by this before the agent sees "
                             "them, so the goal reward becomes ~1. Default: auto from "
                             "--reward (zero=1e8, low=1e4, default=100).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    return parser.parse_args()


def parse_start_pos(raw: str | None, grid_fp: Path) -> tuple[int, int]:
    if raw is not None:
        row, col = raw.split(",")
        return int(row), int(col)

    grid = np.load(grid_fp)
    starts = np.argwhere(grid == 4)
    if len(starts) > 0:
        return int(starts[0][0]), int(starts[0][1])

    empty = np.argwhere(grid == 0)
    if len(empty) == 0:
        raise ValueError(f"No empty start cell found in {grid_fp}")
    return int(empty[0][0]), int(empty[0][1])


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def make_training_start_sampler(grid_fp: Path, fixed_start: tuple[int, int],
                                mode: str, seed: int):
    if mode == "fixed":
        return lambda: fixed_start

    grid = np.load(grid_fp)
    empty_cells = np.argwhere(grid == 0)
    if len(empty_cells) == 0:
        raise ValueError(f"No empty training cells found in {grid_fp}")

    rng = np.random.default_rng(seed)

    def random_start():
        row, col = empty_cells[int(rng.integers(len(empty_cells)))]
        return int(row), int(col)

    return random_start


def train_ppo(agent, env, episodes: int, iters: int, start_sampler):
    agent.set_training(True)
    successes = []

    for episode in trange(episodes, desc="Training PPO"):
        state = env.reset(agent_start_pos=start_sampler())
        agent.new_episode(state)

        reached = False

        for _ in range(iters):
            action = agent.take_action(state)
            state, reward, terminated, info = env.step(action)

            agent.update(state, reward, info["actual_action"])

            if terminated:
                reached = True
                break
        else:
            agent.finish_rollout(state)

        successes.append(1 if reached else 0)

    agent.finish_rollout()
    agent.set_training(False)
    total_successes = int(sum(successes))
    last_100 = successes[-100:] if successes else []
    return {
        "train_total_successes": total_successes,
        "train_total_episodes": int(episodes),
        "train_success_rate": float(total_successes / episodes) if episodes > 0 else 0.0,
        "train_successes_last_100": int(sum(last_100)),
        "train_success_rate_last_100": float(np.mean(last_100)) if last_100 else 0.0,
    }


def evaluate_ppo(agent, Environment, grid_fp, reward_fn, start_pos, sigma,
                 seed, episodes, max_steps, gamma):
    agent.set_training(False)

    successes = []

    for ep in trange(episodes, desc="Evaluating PPO"):
        stats, _, _ = Environment.evaluate_agent(
            grid_fp=grid_fp,
            agent=agent,
            max_steps=max_steps,
            sigma=sigma,
            agent_start_pos=start_pos,
            random_seed=seed + ep,
            reward_fn=reward_fn,
            gamma=gamma,
        )
        successes.append(1 if int(stats.get("targets_remaining", 1)) == 0 else 0)

    total_successes = int(sum(successes))
    return {
        "eval_total_successes": total_successes,
        "eval_total_episodes": int(episodes),
        "eval_success_rate": float(total_successes / episodes) if episodes > 0 else 0.0,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    try:
        from agents.PPO import PPO_agent
        import torch
        from world.environment import (Environment, default_reward_function,
                                       zero_penalty_reward, low_penalty_reward,
                                       make_bfs_distance_reward)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for PPO_agent. Install project dependencies "
                "with `pip install -r requirements.txt`, then rerun this script."
            ) from exc
        if exc.name == "pygame":
            raise SystemExit(
                "pygame is required by world.environment. Install project dependencies "
                "with `pip install -r requirements.txt`, then rerun this script."
            ) from exc
        raise

    if args.reward == "bfs":
        reward_fn, bfs_scale = make_bfs_distance_reward(args.grid)
        if args.reward_scale is None:
            args.reward_scale = bfs_scale
        eval_gamma = 1.0
    else:
        reward_fn = {
            "default": default_reward_function,
            "zero": zero_penalty_reward,
            "low": low_penalty_reward,
        }[args.reward]
        eval_gamma = 0.99 if args.reward == "zero" else 1.0
        if args.reward_scale is None:
            args.reward_scale = REWARD_SCALES[args.reward]

    start_pos = parse_start_pos(args.start_pos, args.grid)
    start_sampler = make_training_start_sampler(
        args.grid,
        fixed_start=start_pos,
        mode=args.train_start_mode,
        seed=args.seed,
    )
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    if args.device.startswith("cuda") and not cuda_available:
        raise SystemExit(
            f"Requested --device {args.device}, but torch.cuda.is_available() is False."
        )

    print(f"Grid: {args.grid}")
    print(f"Start position: {start_pos}")
    print(f"Reward: {args.reward}, sigma={args.sigma}, gamma={args.gamma}")
    print(f"Reward scale: {args.reward_scale:g} (rewards divided by this)")
    print(f"Training start mode: {args.train_start_mode}")
    print(f"Rollout steps: {args.rollout_steps}, update epochs: {args.update_epochs}")
    print(f"Torch device requested: {args.device}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {cuda_device_count}")
    if cuda_available:
        current_cuda_index = torch.cuda.current_device()
        print(f"CUDA current device: {current_cuda_index} ({torch.cuda.get_device_name(current_cuda_index)})")
    print(f"Training: episodes={args.episodes}, max_steps_per_episode={args.iters}")
    print(f"Testing: episodes={args.eval_episodes}, max_steps_per_episode={args.eval_max_steps}")

    env = Environment(
        grid_fp=args.grid,
        no_gui=True,
        sigma=args.sigma,
        agent_start_pos=start_pos,
        target_fps=-1,
        random_seed=args.seed,
        reward_fn=reward_fn,
    )

    agent = PPO_agent(
        grid=args.grid,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        rollout_steps=args.rollout_steps,
        hidden_sizes=hidden_sizes,
        reward_scale=args.reward_scale,
        max_grad_norm=args.max_grad_norm,
        activation=args.activation,
        fourier_freqs=args.fourier_freqs,
        fourier_normalized=not args.fourier_raw,
        seed=args.seed,
        device=args.device,
    )
    actual_device = str(agent.device)
    print(f"Agent actual device: {actual_device}")

    train_metrics = train_ppo(
        agent,
        env,
        args.episodes,
        args.iters,
        start_sampler,
    )

    eval_metrics = evaluate_ppo(
        agent=agent,
        Environment=Environment,
        grid_fp=args.grid,
        reward_fn=reward_fn,
        start_pos=start_pos,
        sigma=args.sigma,
        seed=args.seed,
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        gamma=eval_gamma,
    )

    print("\nPPO result")
    print(f"Training start mode: {args.train_start_mode}")
    print(f"Torch device: {actual_device}")
    print(f"CUDA available: {cuda_available}")
    print(f"Training episodes/max steps: {args.episodes}/{args.iters}")
    print(f"Testing episodes/max steps: {args.eval_episodes}/{args.eval_max_steps}")
    print(f"Training successes: {train_metrics['train_total_successes']}/{train_metrics['train_total_episodes']}")
    print(f"Training success rate: {train_metrics['train_success_rate']:.2f}")
    print(f"Training successes last 100: {train_metrics['train_successes_last_100']}/"
          f"{min(100, train_metrics['train_total_episodes'])}")
    print(f"Train success last 100: {train_metrics['train_success_rate_last_100']:.2f}")
    print(f"Eval successes: {eval_metrics['eval_total_successes']}/{eval_metrics['eval_total_episodes']}")
    print(f"Eval success rate: {eval_metrics['eval_success_rate']:.2f}")


if __name__ == "__main__":
    main()
