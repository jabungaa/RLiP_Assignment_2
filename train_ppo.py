"""PPO training & evaluation.

Importable home for the PPO training pipeline so it is easy to find, call and
reuse (e.g. from `bayesian_search.py` or `train_and_evaluate.py`):

    - `train_ppo(agent, env, ...)`  : run the PPO training loop on a built agent/env.
    - `evaluate_ppo(agent, ...)`    : greedy evaluation, saves a trajectory image.
    - `run_ppo(grid, ...)`          : one-call helper that builds the env + PPO_agent
                                      from hyperparameters, trains, evaluates, and
                                      returns the combined metrics (and optionally
                                      the trained agent).

The `test_ppo.py` CLI imports these functions instead of redefining them.
"""

from __future__ import annotations

from datetime import datetime
from math import radians
from pathlib import Path

import numpy as np
from tqdm import trange

from agents.PPO import PPO_agent
from world.environment_continuous import EnvironmentContinuous

# Goal-reward magnitude of each reward function; used to auto-set reward_scale
# so the agent trains on rewards of magnitude ~1.
REWARD_SCALES = {
    "default": 10.0,   # goal=10, step=-1, collision=-5
    "high": 10.0,      # goal=100000, step=-1, collision=-5
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_start_pos(raw: str | None, grid_fp: Path) -> tuple[int, int]:
    """Returns the start cell (col, row) from a "col,row" string or the grid."""
    if raw is not None:
        col, row = raw.split(",")
        return int(col), int(row)

    grid = np.load(grid_fp)
    starts = np.argwhere(grid == 4)
    if len(starts) > 0:
        return int(starts[0][0]), int(starts[0][1])

    empty = np.argwhere(grid == 0)
    if len(empty) == 0:
        raise ValueError(f"No empty start cell found in {grid_fp}")
    return int(empty[0][0]), int(empty[0][1])


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    """Parses "64,64" into (64, 64)."""
    if not raw.strip():
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def make_training_start_sampler(grid_fp: Path, fixed_start: tuple[int, int],
                                mode: str, seed: int):
    """Builds a callable returning a start cell each episode.

    mode="fixed" always returns `fixed_start`; mode="random" samples a random
    empty cell (seeded).
    """
    if mode == "fixed":
        return lambda: fixed_start

    grid = np.load(grid_fp)
    empty_cells = np.argwhere(grid == 0)
    if len(empty_cells) == 0:
        raise ValueError(f"No empty training cells found in {grid_fp}")

    rng = np.random.default_rng(seed)

    def random_start():
        col, row = empty_cells[int(rng.integers(len(empty_cells)))]
        return int(col), int(row)

    return random_start


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_ppo(agent, env, episodes: int, iters: int, start_sampler,
              train_images_dir=None, greedy_eval_interval=0, greedy_eval_fn=None):
    """Runs the PPO training loop and returns training success metrics."""
    agent.set_training(True)
    successes = []
    consecutive_successes = 0
    next_save_threshold = 20  # first doubling checkpoint after 10
    early_stopped = False
    evaluation_history = []

    for episode in trange(episodes, desc="Training PPO"):
        state = env.reset(agent_start_pos=start_sampler())
        agent.new_episode(state)

        reached = False
        path = [(env.x, env.y)] if train_images_dir is not None else None

        for _ in range(iters):
            action = agent.take_action(state)
            state, reward, terminated, info = env.step(action)

            agent.update(state, reward, info["actual_action"], terminated)

            if path is not None:
                path.append((env.x, env.y))
            if terminated:
                reached = True
                break
        else:
            agent.finish_rollout(state)

        successes.append(1 if reached else 0)

        if reached:
            consecutive_successes += 1
            if train_images_dir is not None:
                if consecutive_successes <= 10 or consecutive_successes >= next_save_threshold:
                    img = env.trajectory_image(path)
                    img.save(train_images_dir / f"episode_{episode:04d}.png")
                    if consecutive_successes >= next_save_threshold:
                        next_save_threshold *= 2
        else:
            consecutive_successes = 0
            next_save_threshold = 20

        if (greedy_eval_interval > 0 and greedy_eval_fn is not None
                and (episode + 1) % greedy_eval_interval == 0):
            evaluation = greedy_eval_fn()
            if isinstance(evaluation, dict):
                evaluation = dict(evaluation)
                should_stop = bool(evaluation.pop("stop_training", False))
                evaluation["training_episode"] = episode + 1
                evaluation_history.append(evaluation)
            else:
                should_stop = bool(evaluation)
            if should_stop:
                print(f"\n[Early stop] Evaluation criterion met at episode {episode + 1}.")
                early_stopped = True
                break

    agent.finish_rollout()
    agent.set_training(False)
    total_successes = int(sum(successes))
    actual_episodes = len(successes)
    last_100 = successes[-100:] if successes else []
    return {
        "train_total_successes": total_successes,
        "train_total_episodes": actual_episodes,
        "train_early_stopped": early_stopped,
        "train_success_rate": float(total_successes / actual_episodes) if actual_episodes > 0 else 0.0,
        "train_successes_last_100": int(sum(last_100)),
        "train_success_rate_last_100": float(np.mean(last_100)) if last_100 else 0.0,
        "evaluation_history": evaluation_history,
    }


def evaluate_ppo(agent, Environment, grid_fp, reward_fn, start_pos, sigma,
                 seed, episodes, max_steps, agent_radius=0.2,
                 move_distance=0.2, turn_angle_deg=15.0,
                 baseline_steps=None, save_image=True):
    """Greedy evaluation over `episodes` via the shared `evaluate_agent`.

    Identical evaluation procedure to the DQN agent; the result is remapped to
    the legacy PPO key names for backward compatibility with existing callers.
    `Environment` is accepted for signature compatibility but the shared
    evaluator always uses `EnvironmentContinuous`.
    """
    from evaluation import evaluate_agent

    res = evaluate_agent(
        agent, grid_fp,
        episodes=episodes,
        max_steps=max_steps,
        sigma=sigma,
        agent_start_pos=start_pos,
        seed=seed,
        reward_fn=reward_fn,
        agent_radius=agent_radius,
        move_distance=move_distance,
        turn_angle_deg=turn_angle_deg,
        optimal_steps=baseline_steps,
        save_image=save_image,
    )
    successes = res.get("eval_success_per_episode", [])
    steps = res.get("eval_steps_per_episode", [])
    successful_steps = [n for n, success in zip(steps, successes) if success]
    within_limit = None
    within_count = None
    spl = None
    if baseline_steps is not None:
        limit = 1.5 * baseline_steps
        within_count = sum(
            1 for n, success in zip(steps, successes) if success and n <= limit
        )
        within_limit = within_count / episodes if episodes else 0.0

        # Aggregate SPL requested by the CLI. For one greedy run this uses that
        # run's step count; for 100 stochastic runs it uses their average. The
        # success-rate factor gives failed evaluations zero contribution.
        if steps:
            spl = (res["eval_success_rate"] * baseline_steps
                   / max(baseline_steps, res["eval_avg_steps"]))

    return {
        "eval_total_successes": res["eval_successes"],
        "eval_total_episodes": res["eval_episodes"],
        "eval_success_rate": res["eval_success_rate"],
        "eval_avg_steps": res["eval_avg_steps"],
        "eval_avg_reward": res["eval_avg_reward"],
        "eval_baseline_steps": baseline_steps,
        "eval_successful_avg_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "eval_within_150pct_baseline_count": within_count,
        "eval_within_150pct_baseline_rate": within_limit,
        "eval_spl": spl,
        "eval_steps": res.get("eval_steps_per_episode", [0])[-1] if res.get("eval_steps_per_episode") else 0,
        "eval_steps_per_episode": steps,
        "eval_success_per_episode": successes,
        "eval_final_pos": res.get("eval_final_pos"),
        "eval_path": res.get("eval_path"),
    }


# --------------------------------------------------------------------------- #
# One-call training+evaluation
# --------------------------------------------------------------------------- #
def run_ppo(
    grid: str | Path,
    *,
    reward: str = "high",
    sigma: float = 0.0,
    seed: int = 1,
    device: str = "cpu",
    episodes: int = 1000,
    iters: int = 1000,
    train_start_mode: str = "random",
    start_pos: tuple[int, int] | None = None,
    eval_episodes: int = 1,
    eval_max_steps: int = 1000,
    # PPO hyperparameters
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    policy_lr: float = 3e-4,
    value_lr: float = 3e-4,
    entropy_coef: float = 0.01,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    replay_capacity: int = 0,
    rollout_steps: int = 4096,
    hidden_sizes: tuple[int, ...] = (128, 128),
    activation: str = "tanh",
    fourier_freqs: int = 0,
    reward_scale: float | None = None,
    max_grad_norm: float = 1e6,
    # environment dynamics
    agent_radius: float = 0.2,
    move_distance: float = 0.2,
    turn_angle_deg: float = 15.0,
    do_eval: bool = True,
    return_agent: bool = False,
):
    """Builds env + PPO_agent from hyperparameters, trains and evaluates.

    Returns the combined train+eval metrics dict (or ``(agent, metrics)`` when
    ``return_agent=True``). This is the single entry point used by both the CLI
    and the Bayesian hyperparameter search.
    """
    grid = Path(grid)
    reward_fn = {
        "default": EnvironmentContinuous._default_reward_function,
        "high": EnvironmentContinuous._high_reward_function,
    }[reward]
    if reward_scale is None:
        reward_scale = REWARD_SCALES[reward]
    if start_pos is None:
        start_pos = parse_start_pos(None, grid)

    start_sampler = make_training_start_sampler(grid, start_pos, train_start_mode, seed)

    env = EnvironmentContinuous(
        grid_fp=grid,
        no_gui=True,
        sigma=sigma,
        agent_start_pos=start_pos,
        target_fps=-1,
        random_seed=seed,
        reward_fn=reward_fn,
        agent_radius=agent_radius,
        move_distance=move_distance,
        turn_angle=radians(turn_angle_deg),
    )

    agent = PPO_agent(
        grid=grid,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        policy_lr=policy_lr,
        value_lr=value_lr,
        entropy_coef=entropy_coef,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        replay_capacity=replay_capacity,
        rollout_steps=rollout_steps,
        hidden_sizes=hidden_sizes,
        reward_scale=reward_scale,
        max_grad_norm=max_grad_norm,
        activation=activation,
        fourier_freqs=fourier_freqs,
        state_size=EnvironmentContinuous.STATE_SIZE,
        seed=seed,
        device=device,
    )

    train_metrics = train_ppo(agent, env, episodes, iters, start_sampler)
    metrics = dict(train_metrics)
    if do_eval:
        eval_metrics = evaluate_ppo(
            agent=agent,
            Environment=EnvironmentContinuous,
            grid_fp=grid,
            reward_fn=reward_fn,
            start_pos=start_pos,
            sigma=sigma,
            seed=seed,
            episodes=eval_episodes,
            max_steps=eval_max_steps,
            agent_radius=agent_radius,
            move_distance=move_distance,
            turn_angle_deg=turn_angle_deg,
        )
        metrics.update(eval_metrics)
    return (agent, metrics) if return_agent else metrics
