"""Standalone PPO test runner.

This script trains and evaluates only `agents.PPO.PPO_agent`. It is intentionally
separate from `new_test.py` and `test_agents.py` so PPO experiments can be run
without also training PI, SARSA, or Monte Carlo agents.

Example:
    python test_ppo.py --grid grid_configs/small_grid.npy --episodes 1000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange

RESULTS_DIR = Path("results/ppo_only")

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
    parser.add_argument("--reward", choices=("default", "zero", "low", "high", "bfs"), default="high")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        help='Torch device to use: "cpu", "cuda", or "cuda:0".')

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--train_start_mode", choices=("random", "fixed"), default="random",
                        help="Training start positions. Evaluation always uses --start_pos.")
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_max_steps", type=int, default=500)
    parser.add_argument("--no_image", action="store_true",
                        help="Do not save the deterministic evaluation path image.")

    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--policy_lr", type=float, default=3e-3)
    parser.add_argument("--value_lr", type=float, default=3e-3)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--hidden_sizes", type=str, default="64,64",
                        help="Comma-separated actor/critic hidden sizes, e.g. 64,64.")
    parser.add_argument("--network_mode", choices=("separate", "shared"), default="separate",
                        help="Use separate actor/critic networks or a shared backbone.")
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
    parser.add_argument("--reward_clip", type=float, default=1000000.0,
                        help="Clip rewards after scaling. With auto reward_scale this "
                             "should rarely bind.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--replay_buffer_size", type=int, default=None,
                        help="Enable off-policy PPO by storing rollouts in a replay buffer "
                             "of this capacity. Each update samples from the full buffer "
                             "instead of only the current rollout. IS correction is "
                             "handled by the existing clipped ratio term.")
    #parser.add_argument("--step_penalty_threshold", type=int, default=None,
    #                    help="Step number within an episode after which each step incurs an extra -10 penalty.")
    #parser.add_argument("--step_penalty", type=float, default=-20.0,
    #                    help="Extra penalty added per step once step_penalty_threshold is exceeded.")
    parser.add_argument("--repeat_visit_penalty", type=float, default=0.0,
                        help="Extra training penalty each time a state is revisited in the same episode.")
    parser.add_argument("--progress_interval", type=int, default=100,
                        help="Window size for the final training progress image.")
    parser.add_argument("--resume_checkpoint", type=Path, default=None,
                        help="Path to a saved PPO checkpoint to continue training from.")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Resume from the latest compatible checkpoint (matching grid, "
                             "network_mode, hidden_sizes, and reward). Off by default: "
                             "every run starts from scratch.")
    parser.add_argument("--fresh_start", action="store_true",
                        help="Deprecated: starting fresh is now the default. "
                             "Kept for backward compatibility.")

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


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_torch_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_ppo_checkpoint(agent, checkpoint_path: Path):
    import torch

    checkpoint = load_torch_checkpoint(checkpoint_path, agent.device)
    metadata = checkpoint.get("metadata", {})

    checkpoint_network_mode = metadata.get("network_mode")
    if checkpoint_network_mode != agent.network_mode:
        raise ValueError(
            "Checkpoint network_mode does not match this run. "
            f"Checkpoint has {checkpoint_network_mode!r}, current run has "
            f"{agent.network_mode!r}. Use --network_mode {checkpoint_network_mode}."
        )

    checkpoint_hidden_sizes = tuple(metadata.get("hidden_sizes", ()))
    if checkpoint_hidden_sizes != tuple(agent.hidden_sizes):
        raise ValueError(
            "Checkpoint hidden_sizes does not match this run. "
            f"Checkpoint has {checkpoint_hidden_sizes}, current run has "
            f"{tuple(agent.hidden_sizes)}. Use --hidden_sizes "
            f"{','.join(str(size) for size in checkpoint_hidden_sizes)}."
        )

    if agent.network_mode == "shared":
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    if "agent_rng_state" in checkpoint:
        agent.rng.bit_generator.state = checkpoint["agent_rng_state"]
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    agent._clear_rollout()
    return metadata


def latest_compatible_checkpoint(
    results_dir: Path,
    grid: Path,
    network_mode: str,
    hidden_sizes: tuple[int, ...],
    reward: str,
):
    candidates = sorted(
        results_dir.glob("*_checkpoint.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for checkpoint_path in candidates:
        try:
            checkpoint = load_torch_checkpoint(checkpoint_path, "cpu")
        except Exception:
            continue

        metadata = checkpoint.get("metadata", {})
        if metadata.get("grid") != str(grid):
            continue
        if metadata.get("network_mode") != network_mode:
            continue
        if tuple(metadata.get("hidden_sizes", ())) != tuple(hidden_sizes):
            continue
        checkpoint_reward = metadata.get(
            "reward", metadata.get("args", {}).get("reward")
        )
        if checkpoint_reward != reward:
            continue
        return checkpoint_path

    return None


def save_ppo_checkpoint(
    agent,
    checkpoint_path: Path,
    args,
    train_metrics: dict,
    eval_metrics: dict,
    image_metrics: dict,
):
    import torch

    checkpoint = {
        "metadata": {
            "agent": "PPO_agent",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "grid": str(args.grid),
            "reward": args.reward,
            "reward_scale": float(args.reward_scale),
            "network_mode": agent.network_mode,
            "hidden_sizes": list(agent.hidden_sizes),
            "input_dim": int(agent.input_dim),
            "device": str(agent.device),
            "args": json_safe(vars(args)),
            "train_metrics": json_safe(train_metrics),
            "eval_metrics": json_safe(eval_metrics),
            "image_metrics": json_safe(image_metrics),
        },
        "agent_rng_state": agent.rng.bit_generator.state,
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }

    if agent.network_mode == "shared":
        checkpoint.update({
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        })
    else:
        checkpoint.update({
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        })

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def bfs_shortest_path(grid_fp: Path, start: tuple[int, int]) -> int | None:
    grid = np.load(grid_fp)
    targets = np.argwhere(grid == 3)
    if len(targets) == 0:
        raise ValueError(f"No target cell with value 3 found in {grid_fp}")

    target = (int(targets[0][0]), int(targets[0][1]))
    if start == target:
        return 0

    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    visited[start] = True
    queue = deque([(start, 0)])

    while queue:
        (row, col), distance = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc] or grid[nr, nc] in (1, 2):
                continue
            if (nr, nc) == target:
                return distance + 1
            visited[nr, nc] = True
            queue.append(((nr, nc), distance + 1))

    return None


ACTION_DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))


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


def save_training_progress_image(successes: list[int], out_path: Path,
                                 interval: int = 100):
    width, height = 900, 520
    margin_left, margin_right = 70, 30
    margin_top, margin_bottom = 45, 65
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    n = len(successes)
    if n == 0:
        image.save(out_path)
        return {"training_progress_image_path": str(out_path)}

    cumulative = np.cumsum(successes)
    xs = np.arange(1, n + 1)

    window_ends = list(range(interval, n + 1, interval)) if interval > 0 else []
    if not window_ends or window_ends[-1] != n:
        window_ends.append(n)
    window_successes = [
        int(sum(successes[max(0, end - interval):end])) if interval > 0 else int(sum(successes))
        for end in window_ends
    ]

    y_max = max(1, int(cumulative[-1]), max(window_successes) if window_successes else 1)

    def xy(ep: float, value: float):
        x = margin_left + ((ep - 1) / max(1, n - 1)) * plot_w
        y = margin_top + (1.0 - value / y_max) * plot_h
        return x, y

    draw.text((margin_left, 14), "PPO training success progress", fill=(0, 0, 0, 255))
    draw.text((margin_left, height - 30), "episode", fill=(0, 0, 0, 255))
    draw.text((8, margin_top + plot_h // 2 - 10), "successes", fill=(0, 0, 0, 255))

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(0, 0, 0, 255), width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(0, 0, 0, 255), width=2)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y_max * frac
        _x, y = xy(1, value)
        draw.line((margin_left - 5, y, margin_left + plot_w, y), fill=(230, 230, 230, 255), width=1)
        draw.text((18, y - 7), str(int(round(value))), fill=(0, 0, 0, 255))

    if n > 1:
        points = [xy(float(ep), float(val)) for ep, val in zip(xs, cumulative)]
        draw.line(points, fill=(25, 90, 210, 255), width=3)

    bar_width = max(2, int(plot_w / max(1, len(window_ends)) * 0.55))
    for end, value in zip(window_ends, window_successes):
        x, y = xy(end, value)
        baseline_y = margin_top + plot_h
        draw.rectangle(
            (x - bar_width / 2, y, x + bar_width / 2, baseline_y),
            fill=(244, 132, 66, 150),
            outline=(210, 95, 35, 255),
        )

    draw.line((width - 260, 26, width - 230, 26), fill=(25, 90, 210, 255), width=3)
    draw.text((width - 224, 18), "cumulative successes", fill=(0, 0, 0, 255))
    draw.rectangle((width - 260, 48, width - 230, 65), fill=(244, 132, 66, 150), outline=(210, 95, 35, 255))
    draw.text((width - 224, 47), f"successes / {interval} eps", fill=(0, 0, 0, 255))
    draw.text((margin_left, height - 52), f"episodes={n}, total_successes={int(cumulative[-1])}", fill=(0, 0, 0, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {"training_progress_image_path": str(out_path)}


def train_ppo(agent, env, episodes: int, iters: int, start_sampler,
              repeat_visit_penalty: float):
    agent.set_training(True)
    episode_returns = []
    episode_learning_returns = []
    episode_steps = []
    successes = []
    visit_counts = np.zeros_like(np.load(env.grid_fp), dtype=np.int64)

    for episode in trange(episodes, desc="Training PPO"):
        state = env.reset(agent_start_pos=start_sampler())
        agent.new_episode(state)
        visit_counts[state] += 1
        episode_visit_counts = {state: 1}

        total_reward = 0.0
        total_learning_reward = 0.0
        reached = False
        steps = 0

        for _ in range(iters):
            action = agent.take_action(state)
            state, reward, terminated, info = env.step(action)
            revisits = episode_visit_counts.get(state, 0)
            loop_penalty = repeat_visit_penalty * revisits
            learning_reward = reward - loop_penalty

            agent.update(state, learning_reward, info["actual_action"])
            visit_counts[state] += 1
            episode_visit_counts[state] = revisits + 1

            total_reward += reward
            total_learning_reward += learning_reward
            steps += 1
            if terminated:
                reached = True
                break
        else:
            agent.finish_rollout(state)

        episode_returns.append(float(total_reward))
        episode_learning_returns.append(float(total_learning_reward))
        episode_steps.append(int(steps))
        successes.append(1 if reached else 0)

    agent.finish_rollout()
    agent.set_training(False)
    total_successes = int(sum(successes))
    return {
        "train_total_successes": total_successes,
        "train_total_episodes": int(episodes),
        "train_success_rate": float(total_successes / episodes) if episodes > 0 else 0.0,
        "train_avg_return_last_100": float(np.mean(episode_returns[-100:])),
        "train_avg_learning_return_last_100": float(np.mean(episode_learning_returns[-100:])),
        "train_success_rate_last_100": float(np.mean(successes[-100:])),
        "train_avg_steps_last_100": float(np.mean(episode_steps[-100:])),
        "train_visit_counts": visit_counts,
        "train_successes_by_episode": successes,
    }


def evaluate_ppo(agent, Environment, grid_fp, reward_fn, start_pos, sigma,
                 seed, episodes, max_steps, gamma):
    old_epsilon = getattr(agent, "epsilon", None)
    if old_epsilon is not None:
        agent.epsilon = 0.0
    agent.set_training(False)

    rewards = []
    steps = []
    failed_moves = []
    agent_moves = []
    successes = []
    times = []

    for ep in trange(episodes, desc="Evaluating PPO"):
        start_time = time.perf_counter()
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
        times.append(time.perf_counter() - start_time)
        rewards.append(float(stats["cumulative_reward"]))
        steps.append(int(stats["total_steps"]))
        failed_moves.append(int(stats["total_failed_moves"]))
        agent_moves.append(int(stats["total_agent_moves"]))
        successes.append(1 if int(stats.get("targets_remaining", 1)) == 0 else 0)

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    return {
        "eval_avg_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_avg_steps": float(np.mean(steps)),
        "eval_avg_failed_moves": float(np.mean(failed_moves)),
        "eval_avg_agent_moves": float(np.mean(agent_moves)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_avg_time": float(np.mean(times)),
        "eval_rewards": rewards,
        "eval_steps": steps,
    }


def save_path_image(agent, Environment, visualize_path, grid_fp, reward_fn,
                    start_pos, seed, max_steps, gamma, out_path: Path):
    old_epsilon = getattr(agent, "epsilon", None)
    if old_epsilon is not None:
        agent.epsilon = 0.0
    agent.set_training(False)

    stats, _elapsed, agent_path = Environment.evaluate_agent(
        grid_fp=grid_fp,
        agent=agent,
        max_steps=max_steps,
        sigma=0.0,
        agent_start_pos=start_pos,
        random_seed=seed,
        reward_fn=reward_fn,
        gamma=gamma,
    )

    grid_cells = np.load(grid_fp)
    path_image = visualize_path(grid_cells, agent_path)
    path_image.save(out_path)

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    return {
        "image_path": str(out_path),
        "image_total_steps": int(stats["total_steps"]),
        "image_total_failed_moves": int(stats["total_failed_moves"]),
        "image_total_agent_moves": int(stats["total_agent_moves"]),
        "image_reached_goal": int(stats.get("targets_remaining", 1)) == 0,
    }


def save_policy_entropy_image(agent, grid_fp: Path, start_pos: tuple[int, int],
                              out_path: Path):
    old_epsilon = getattr(agent, "epsilon", None)
    if old_epsilon is not None:
        agent.epsilon = 0.0
    agent.set_training(False)

    grid = np.load(grid_fp)
    scalar = 30
    image_size = tuple((g * scalar) + 2 for g in grid.shape)

    wall_colors = {
        1: (57, 57, 57, 255),
        2: (57, 57, 57, 255),
        3: (34, 139, 34, 255),
    }
    entropy_grid = np.full(grid.shape, np.nan, dtype=float)
    max_entropy = math.log(4)

    image = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    for col in range(grid.shape[0]):
        for row in range(grid.shape[1]):
            if grid[col, row] not in (0, 4):
                continue
            probs = agent.policy((col, row))
            valid_probs = probs[probs > 0]
            entropy_grid[col, row] = -float(np.sum(valid_probs * np.log(valid_probs)))

    for row in range(grid.shape[1]):
        y = row * scalar + 1
        for col in range(grid.shape[0]):
            x = col * scalar + 1
            value = int(grid[col, row])
            fill = wall_colors.get(value, (255, 255, 255, 255))

            if value in (0, 4) and not np.isnan(entropy_grid[col, row]):
                ratio = min(1.0, entropy_grid[col, row] / max_entropy) if max_entropy > 0 else 0.0
                red = int(255 * ratio)
                green = int(230 * (1.0 - ratio))
                blue = int(255 * (1.0 - ratio))
                fill = (red, green, blue, 255)

            if (col, row) == start_pos:
                fill = (242, 211, 82, 255)

            draw.rectangle(
                (x, y, x + scalar, y + scalar),
                fill=fill,
                outline=(220, 220, 220, 255),
            )

            if value not in (0, 4):
                continue

            probs = agent.policy((col, row))
            action = int(np.argmax(probs))
            dx, dy = ACTION_DELTAS[action]

            center_x = col * scalar + scalar // 2
            center_y = row * scalar + scalar // 2
            end_x = center_x + dx * int(scalar * 0.32)
            end_y = center_y + dy * int(scalar * 0.32)

            draw.line(
                (center_x, center_y, end_x, end_y),
                fill=(25, 90, 210, 255),
                width=2,
            )

            angle = math.atan2(end_y - center_y, end_x - center_x)
            head_len = 5
            for offset in (math.pi * 0.78, -math.pi * 0.78):
                hx = end_x + head_len * math.cos(angle + offset)
                hy = end_y + head_len * math.sin(angle + offset)
                draw.line((end_x, end_y, hx, hy), fill=(25, 90, 210, 255), width=2)

            if not np.isnan(entropy_grid[col, row]):
                text = f"{entropy_grid[col, row]:.2f}"
                bbox = draw.textbbox((0, 0), text)
                tw = bbox[2] - bbox[0]
                draw.text(
                    (x + (scalar - tw) / 2, y + 2),
                    text,
                    fill=(0, 0, 0, 255),
                )

    image.save(out_path)

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    finite_entropy = entropy_grid[~np.isnan(entropy_grid)]
    return {
        "policy_entropy_image_path": str(out_path),
        "policy_entropy_mean": float(np.mean(finite_entropy)) if len(finite_entropy) else None,
        "policy_entropy_min": float(np.min(finite_entropy)) if len(finite_entropy) else None,
        "policy_entropy_max": float(np.max(finite_entropy)) if len(finite_entropy) else None,
    }


def save_policy_probs_image(agent, grid_fp: Path, start_pos: tuple[int, int],
                            out_path: Path):
    """Large image: all four action probabilities per cell in compass layout.

    Each cell is 120×120 px.  For every direction the probability is printed
    as a large number at the cell edge and a proportional arrow is drawn from
    the centre.  Dominant action is blue; the others are light grey.
    """
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        try:
            font = ImageFont.load_default(size=20)
        except Exception:
            font = ImageFont.load_default()

    old_epsilon = getattr(agent, "epsilon", None)
    if old_epsilon is not None:
        agent.epsilon = 0.0
    agent.set_training(False)

    grid   = np.load(grid_fp)
    scalar = 120
    pad    = 2
    width  = grid.shape[0] * scalar + 2 * pad
    height = grid.shape[1] * scalar + 2 * pad

    wall_colors = {
        1: (57,  57,  57,  255),
        2: (57,  57,  57,  255),
        3: (34,  139, 34,  255),
    }
    COLOR_BEST = (25,  90,  210, 255)
    COLOR_REST = (180, 180, 180, 255)

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw  = ImageDraw.Draw(image)

    # ACTION_DELTAS: 0→(dr=0,dc=+1) down, 1→(dr=0,dc=-1) up,
    #                2→(dr=-1,dc=0) left,  3→(dr=+1,dc=0) right  (in image coords)
    MAX_ARROW = scalar // 2 - 10   # max shaft length in px

    for col in range(grid.shape[0]):
        for row in range(grid.shape[1]):
            cx = pad + col * scalar
            cy = pad + row * scalar
            value = int(grid[col, row])
            fill  = wall_colors.get(value, (250, 250, 250, 255))
            if (col, row) == start_pos:
                fill = (242, 211, 82, 255)
            draw.rectangle((cx, cy, cx + scalar - 1, cy + scalar - 1),
                            fill=fill, outline=(180, 180, 180, 255), width=1)

            if value not in (0, 4):
                continue

            probs    = agent.policy((col, row))
            best_a   = int(np.argmax(probs))
            center_x = cx + scalar // 2
            center_y = cy + scalar // 2

            for action, (dr, dc) in enumerate(ACTION_DELTAS):
                p     = float(probs[action])
                color = COLOR_BEST if action == best_a else COLOR_REST
                lw    = 5 if action == best_a else 3

                # --- arrow: minimum visible length so even low-prob arrows show ---
                shaft = max(16, int(MAX_ARROW * p))
                ex    = center_x + dr * shaft
                ey    = center_y + dc * shaft
                draw.line((center_x, center_y, ex, ey), fill=color, width=lw)
                angle   = math.atan2(ey - center_y, ex - center_x)
                head_ln = max(10, int(shaft * 0.45))
                for off in (math.pi * 0.75, -math.pi * 0.75):
                    hx = ex + head_ln * math.cos(angle + off)
                    hy = ey + head_ln * math.sin(angle + off)
                    draw.line((ex, ey, hx, hy), fill=color, width=lw)

                # --- probability number, hugging the cell edge ---
                text = f"{p:.2f}"
                bb   = draw.textbbox((0, 0), text, font=font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]

                margin = 5
                if dc == -1:    # up
                    tx = cx + (scalar - tw) // 2
                    ty = cy + margin
                elif dc == 1:   # down
                    tx = cx + (scalar - tw) // 2
                    ty = cy + scalar - th - margin
                elif dr == -1:  # left
                    tx = cx + margin
                    ty = cy + (scalar - th) // 2
                else:           # right
                    tx = cx + scalar - tw - margin
                    ty = cy + (scalar - th) // 2

                # white halo so number is readable over arrow
                for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.text((tx + ox, ty + oy), text,
                              fill=(255, 255, 255, 200), font=font)
                draw.text((tx, ty), text, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    return {"policy_probs_image_path": str(out_path)}


def save_training_visits_image(grid_fp: Path, visit_counts: np.ndarray, out_path: Path):
    grid = np.load(grid_fp)
    scalar = 30
    image_size = tuple((g * scalar) + 2 for g in grid.shape)
    image = Image.new("RGBA", image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    colors = {
        0: (255, 255, 255, 255),
        1: (57, 57, 57, 255),
        2: (57, 57, 57, 255),
        3: (34, 139, 34, 255),
        4: (242, 211, 82, 255),
    }
    max_visits = int(visit_counts.max())

    for row in range(grid.shape[1]):
        y = row * scalar + 1
        for col in range(grid.shape[0]):
            x = col * scalar + 1
            value = int(grid[col, row])
            fill = colors.get(value, (255, 255, 255, 255))

            if value in (0, 3, 4) and max_visits > 0:
                intensity = visit_counts[col, row] / max_visits
                red = 255
                green_blue = int(255 * (1.0 - intensity))
                fill = (red, green_blue, green_blue, 255)

            draw.rectangle(
                (x, y, x + scalar, y + scalar),
                fill=fill,
                outline=(220, 220, 220, 255),
            )

            if value in (0, 3, 4) and visit_counts[col, row] > 0:
                text = str(int(visit_counts[col, row]))
                bbox = draw.textbbox((0, 0), text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text(
                    (x + (scalar - tw) / 2, y + (scalar - th) / 2),
                    text,
                    fill=(0, 0, 0, 255),
                )

    image.save(out_path)
    return {"training_visits_image_path": str(out_path)}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from agents.PPO import PPO_agent
        import torch
        from world.environment_continuous import EnvironmentContinuous
        from world.path_visualizer import visualize_path
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


    reward_fn = {
        "default": EnvironmentContinuous._default_reward_function,
        "high": EnvironmentContinuous._high_reward_function,
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
    auto_resume_used = False
    if args.resume_checkpoint is None and args.auto_resume:
        args.resume_checkpoint = latest_compatible_checkpoint(
            RESULTS_DIR,
            grid=args.grid,
            network_mode=args.network_mode,
            hidden_sizes=hidden_sizes,
            reward=args.reward,
        )
        auto_resume_used = args.resume_checkpoint is not None
        if args.resume_checkpoint is None:
            print("Auto-resume requested, but no compatible checkpoint found.")

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    if args.device.startswith("cuda") and not cuda_available:
        raise SystemExit(
            f"Requested --device {args.device}, but torch.cuda.is_available() is False."
        )

    print(f"Grid: {args.grid}")
    print(f"Start position: {start_pos}")
    print(f"Reward: {args.reward}, sigma={args.sigma}, gamma={args.gamma}")
    print(f"Reward scale: {args.reward_scale:g} (rewards divided by this), "
          f"clip after scaling: {args.reward_clip:g}")
    print(f"Training start mode: {args.train_start_mode}")
    print(f"Network mode: {args.network_mode}")
    print(f"Rollout steps: {args.rollout_steps}, minibatch: {args.minibatch_size}, "
          f"update epochs: {args.update_epochs}")
    print(f"Torch device requested: {args.device}")
    if args.resume_checkpoint is not None:
        resume_kind = "auto" if auto_resume_used else "manual"
        print(f"Resume checkpoint ({resume_kind}): {args.resume_checkpoint}")
    else:
        print("Resume checkpoint: none (fresh start)")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {cuda_device_count}")
    if cuda_available:
        current_cuda_index = torch.cuda.current_device()
        print(f"CUDA current device: {current_cuda_index} ({torch.cuda.get_device_name(current_cuda_index)})")
    print(f"Repeat visit penalty: {args.repeat_visit_penalty}")
    print(f"Training: episodes={args.episodes}, max_steps_per_episode={args.iters}")
    print(f"Testing: episodes={args.eval_episodes}, max_steps_per_episode={args.eval_max_steps}")

    env = EnvironmentContinuous(
        grid_fp=args.grid,
        no_gui=True,
        sigma=args.sigma,
        agent_start_pos=start_pos,
        target_fps=-1,
        random_seed=args.seed,
        reward_fn=reward_fn,
        #step_penalty_threshold=args.step_penalty_threshold,
        #step_penalty=args.step_penalty,
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
        minibatch_size=args.minibatch_size,
        hidden_sizes=hidden_sizes,
        network_mode=args.network_mode,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        max_grad_norm=args.max_grad_norm,
        replay_buffer_size=args.replay_buffer_size,
        activation=args.activation,
        fourier_freqs=args.fourier_freqs,
        fourier_normalized=not args.fourier_raw,
        seed=args.seed,
        device=args.device,
    )
    actual_device = str(agent.device)
    print(f"Agent actual device: {actual_device}")

    stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    resumed_from = None
    if args.resume_checkpoint is not None:
        resume_metadata = load_ppo_checkpoint(agent, args.resume_checkpoint)
        resumed_from = str(args.resume_checkpoint)
        print(
            "Loaded checkpoint "
            f"from {args.resume_checkpoint} "
            f"(created_at={resume_metadata.get('created_at')})"
        )

    if not args.no_image:
        init_entropy_path = RESULTS_DIR / f"{stamp}_init_policy_entropy.png"
        init_probs_path   = RESULTS_DIR / f"{stamp}_init_policy_probs.png"
        init_stats = save_policy_entropy_image(
            agent=agent, grid_fp=args.grid,
            start_pos=start_pos, out_path=init_entropy_path,
        )
        save_policy_probs_image(
            agent=agent, grid_fp=args.grid,
            start_pos=start_pos, out_path=init_probs_path,
        )
        print(
            f"Initial policy entropy — "
            f"mean={init_stats['policy_entropy_mean']:.4f}  "
            f"min={init_stats['policy_entropy_min']:.4f}  "
            f"max={init_stats['policy_entropy_max']:.4f}  "
            f"(max possible={math.log(4):.4f})"
        )
        print(f"Initial policy entropy image : {init_entropy_path}")
        print(f"Initial policy probs image   : {init_probs_path}")

    train_start = time.perf_counter()
    train_metrics = train_ppo(
        agent,
        env,
        args.episodes,
        args.iters,
        start_sampler,
        args.repeat_visit_penalty,
    )
    train_time = time.perf_counter() - train_start
    visit_counts = train_metrics.pop("train_visit_counts")
    successes_by_episode = train_metrics.pop("train_successes_by_episode")

    eval_metrics = evaluate_ppo(
        agent=agent,
        Environment=EnvironmentContinuous,
        grid_fp=args.grid,
        reward_fn=reward_fn,
        start_pos=start_pos,
        sigma=args.sigma,
        seed=args.seed,
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        gamma=eval_gamma,
    )

    image_metrics = {}
    if not args.no_image:
        image_path = RESULTS_DIR / f"{stamp}_path.png"
        visits_image_path = RESULTS_DIR / f"{stamp}_training_visits.png"
        progress_image_path = RESULTS_DIR / f"{stamp}_training_progress.png"
        entropy_image_path = RESULTS_DIR / f"{stamp}_policy_entropy.png"
        probs_image_path = RESULTS_DIR / f"{stamp}_policy_probs.png"
        image_metrics = save_path_image(
            agent=agent,
            Environment=EnvironmentContinuous,
            visualize_path=visualize_path,
            grid_fp=args.grid,
            reward_fn=reward_fn,
            start_pos=start_pos,
            seed=args.seed,
            max_steps=args.eval_max_steps,
            gamma=eval_gamma,
            out_path=image_path,
        )
        image_metrics.update(save_training_visits_image(
            grid_fp=args.grid,
            visit_counts=visit_counts,
            out_path=visits_image_path,
        ))
        image_metrics.update(save_policy_entropy_image(
            agent=agent,
            grid_fp=args.grid,
            start_pos=start_pos,
            out_path=entropy_image_path,
        ))
        image_metrics.update(save_policy_probs_image(
            agent=agent,
            grid_fp=args.grid,
            start_pos=start_pos,
            out_path=probs_image_path,
        ))
        image_metrics.update(save_training_progress_image(
            successes_by_episode,
            progress_image_path,
            args.progress_interval,
        ))

    checkpoint_path = RESULTS_DIR / f"{stamp}_checkpoint.pt"
    save_ppo_checkpoint(
        agent=agent,
        checkpoint_path=checkpoint_path,
        args=args,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        image_metrics=image_metrics,
    )

    result = {
        "agent": "PPO_agent",
        "grid": str(args.grid),
        "start_pos": list(start_pos),
        "reward": args.reward,
        "reward_scale": float(args.reward_scale),
        "reward_clip": float(args.reward_clip),
        "gamma": args.gamma,
        "policy_lr": args.policy_lr,
        "entropy_coef": args.entropy_coef,
        "sigma": args.sigma,
        "seed": args.seed,
        "device_requested": args.device,
        "device_actual": actual_device,
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(cuda_device_count),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "resume_checkpoint": resumed_from,
        "auto_resume_used": bool(auto_resume_used),
        "checkpoint_path": str(checkpoint_path),
        "episodes": args.episodes,
        "iters": args.iters,
        "train_start_mode": args.train_start_mode,
        "eval_episodes": args.eval_episodes,
        "eval_max_steps": args.eval_max_steps,
        "hidden_sizes": list(hidden_sizes),
        "network_mode": args.network_mode,
        "rollout_steps": args.rollout_steps,
        "minibatch_size": args.minibatch_size,
        "update_epochs": args.update_epochs,
        "repeat_visit_penalty": args.repeat_visit_penalty,
        "progress_interval": args.progress_interval,
        "train_time_sec": float(train_time),
        **train_metrics,
        **eval_metrics,
        **image_metrics,
    }

    out_path = RESULTS_DIR / f"{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nPPO result")
    print(f"Training start mode: {args.train_start_mode}")
    print(f"Network mode: {args.network_mode}")
    print(f"Torch device: {actual_device}")
    print(f"CUDA available: {cuda_available}")
    print(f"Repeat visit penalty: {args.repeat_visit_penalty}")
    print(f"Training episodes/max steps: {args.episodes}/{args.iters}")
    print(f"Testing episodes/max steps: {args.eval_episodes}/{args.eval_max_steps}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Training successes: {train_metrics['train_total_successes']}/{train_metrics['train_total_episodes']}")
    print(f"Training success rate: {train_metrics['train_success_rate']:.2f}")
    print(f"Train success last 100: {train_metrics['train_success_rate_last_100']:.2f}")
    print(f"Eval success rate: {eval_metrics['eval_success_rate']:.2f}")
    print(f"Eval avg reward: {eval_metrics['eval_avg_reward']:.2f}")
    print(f"Eval avg steps: {eval_metrics['eval_avg_steps']:.1f}")
    print(f"Checkpoint: {checkpoint_path}")
    if image_metrics:
        print(f"Path image: {image_metrics['image_path']}")
        print(f"Training visits image: {image_metrics['training_visits_image_path']}")
        print(f"Policy + entropy image: {image_metrics['policy_entropy_image_path']}")
        if "training_progress_image_path" in image_metrics:
            print(f"Training progress image: {image_metrics['training_progress_image_path']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
