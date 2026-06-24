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

import copy
from datetime import datetime
from math import radians
import os
from pathlib import Path

import numpy as np
import random
import torch
from tqdm import trange

from agents.PPO import PPO_agent
from world.environment_continuous import EnvironmentContinuous

# Goal-reward magnitude of each reward function. Used to auto-set reward_scale
# so the success reward becomes ~1 after the PPO agent divides every reward by
# it -- this keeps the critic's value targets at order ~1 for stable training.
# Each value MUST equal the goal reward returned by the matching reward fn in
# EnvironmentContinuous, otherwise the normalisation is wrong.
REWARD_SCALES = {
    "default": 1.0,      # _default_reward_function: goal=+1, collision=-0.25, step=-0.001/-0.01
    "high": 1000.0,      # _high_reward_function:    goal=+1000, collision=-5, move=+0.1, step=-1
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

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  # controls Python hash randomness


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def ppo_train_metrics(history: list[dict]) -> dict:
    """Aggregates per-episode training history into success-rate metrics.

    Mirrors the metrics produced for the DQN training history.
    """
    successes = [1 if ep["terminated"] else 0 for ep in history]
    n = len(successes)
    last_100 = successes[-100:]
    total = int(sum(successes))
    return {
        "train_total_successes": total,
        "train_total_episodes": n,
        "train_success_rate": float(total / n) if n else 0.0,
        "train_successes_last_100": int(sum(last_100)),
        "train_success_rate_last_100": float(np.mean(last_100)) if last_100 else 0.0,
    }


def train_ppo(agent, env, *, max_steps_total: int, max_steps_per_episode: int = 1000,
              short_train_steps_eval: int | None = None,
              mid_train_steps_eval: int | None = None,
              start_pos: tuple[int, int] | None = None, start_sampler=None,
              train_images_dir=None, greedy_eval_interval=0, greedy_eval_fn=None,
              seed: int = 0):
    """Step-budgeted PPO training loop, consistent with `train_dqn.train_DQN`.

    Trains until `max_steps_total` environment steps, snapshotting the agent at
    the short/mid step thresholds (for sample-efficiency / AUC analysis). The
    per-episode start cell comes from `start_sampler()` if given (random-start
    curriculum), else the fixed `start_pos`, else the environment default.

    Returns:
        (agent, training_history, short_train_agent, mid_train_agent) -- the
        same 4-tuple shape `train_DQN` returns.
    """
    set_all_seeds(seed)
    agent.set_training(True)
    training_history = []
    step_count = 0
    short_train_agent = None
    mid_train_agent = None
    consecutive_successes = 0
    next_save_threshold = 20
    episode = 0
    print(f"Training PPO agent for a maximum of {max_steps_total} steps...")

    while step_count < max_steps_total:
        start = start_sampler() if start_sampler is not None else start_pos
        state = env.reset(agent_start_pos=start)
        agent.new_episode(state)

        terminated = False
        total_reward = 0.0
        path = [(env.x, env.y)] if train_images_dir is not None else None

        for step in range(max_steps_per_episode):
            action = agent.take_action(state)
            new_state, reward, terminated, info = env.step(action)
            agent.update(state, reward, info["actual_action"], terminated)
            state = new_state
            if path is not None:
                path.append((env.x, env.y))
            total_reward += reward
            step_count += 1
            if terminated:
                break
            if step_count % 10000 == 0:
                print(f"Step {step_count}/{max_steps_total}, Episode {episode}")
            if short_train_agent is None and short_train_steps_eval and step_count >= short_train_steps_eval:
                short_train_agent = copy.deepcopy(agent)
            if mid_train_agent is None and mid_train_steps_eval and step_count >= mid_train_steps_eval:
                mid_train_agent = copy.deepcopy(agent)
            if step_count >= max_steps_total:
                break
        else:
            # Episode truncated (no terminal): bootstrap the GAE chain.
            agent.finish_rollout(state)

        training_history.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": step + 1,
            "terminated": terminated,
            "targets_reached": env.world_stats.get("total_targets_reached", 0),
            "failed_moves": env.world_stats.get("total_failed_moves", 0),
        })

        # Optional training trajectory images (doubling cadence on success).
        if train_images_dir is not None and terminated and path is not None:
            consecutive_successes += 1
            if consecutive_successes <= 10 or consecutive_successes >= next_save_threshold:
                env.trajectory_image(path).save(train_images_dir / f"episode_{episode:04d}.png")
                if consecutive_successes >= next_save_threshold:
                    next_save_threshold *= 2
        elif train_images_dir is not None:
            consecutive_successes = 0
            next_save_threshold = 20

        # Optional periodic greedy evaluation with early stop.
        if (greedy_eval_interval > 0 and greedy_eval_fn is not None
                and (episode + 1) % greedy_eval_interval == 0):
            evaluation = greedy_eval_fn()
            should_stop = (bool(evaluation.get("stop_training", False))
                           if isinstance(evaluation, dict) else bool(evaluation))
            if should_stop:
                print(f"\n[Early stop] Evaluation criterion met at episode {episode + 1}.")
                break

        episode += 1

    # agent.finish_rollout()
    # agent.set_training(False)
    return agent, training_history, short_train_agent, mid_train_agent


def evaluate_ppo(
    agent,
    grid: str | Path,
    max_steps_per_episode: int = 1000,
    sigma: float = 0.0,
    agent_start_pos: tuple[int, int] | None = None,
    no_gui: bool = True,
    random_seed: int = 0,
    move_distance: float = 0.2,
    episodes: int = 100,
    reward_fn=None,
    optimal_steps: int | None = None,
    agent_radius: float = 0.5,
    turn_angle_deg: float = 15.0,
):
    """Greedy evaluation of a PPO agent via the shared `evaluate_agent`.

    Thin wrapper, identical in shape to `train_dqn.evaluate_DQN`: same procedure,
    same parameters, same returned keys -- so the two agents are evaluated and
    reported uniformly.
    """
    from evaluation import evaluate_agent

    res = evaluate_agent(
        agent, grid,
        episodes=episodes,
        max_steps=max_steps_per_episode,
        sigma=sigma,
        agent_start_pos=agent_start_pos,
        seed=random_seed,
        reward_fn=reward_fn,
        agent_radius=agent_radius,
        move_distance=move_distance,
        turn_angle_deg=turn_angle_deg,
        optimal_steps=optimal_steps,
    )
    return {
        "eval_success_rate": res["eval_success_rate"],
        "SPL": res["eval_avg_spl"] if res["eval_avg_spl"] is not None else 0.0,
        "total_reward": res["eval_avg_reward"],
        "avg_steps": res["eval_avg_steps"],
        "avg_failed_moves": res["eval_avg_failed_moves"],
    }


# --------------------------------------------------------------------------- #
# One-call training+evaluation
# --------------------------------------------------------------------------- #
def run_ppo(
    grid: str | Path,
    *,
    reward: str = "high",
    sigma: float = 0.05,
    seed: int = 1,
    device: str = "cpu",
    episodes: int = 10000,
    iters: int = 1000,
    train_start_mode: str = "fixed",
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
        # PPO_agent is on-policy only now; replay_capacity is no longer a param.
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

    # Episode budget -> step budget (max_steps_total = episodes x iters) for the
    # step-budgeted trainer; random/fixed starts via the sampler.
    agent, history, _short, _mid = train_ppo(
        agent, env,
        max_steps_total=episodes * iters,
        max_steps_per_episode=iters,
        start_sampler=start_sampler,
        seed=seed,
    )
    metrics = ppo_train_metrics(history)
    if do_eval:
        eval_metrics = evaluate_ppo(
            agent, grid,
            max_steps_per_episode=eval_max_steps,
            sigma=sigma,
            agent_start_pos=start_pos,
            random_seed=seed,
            move_distance=move_distance,
            episodes=eval_episodes,
            reward_fn=reward_fn,
            agent_radius=agent_radius,
            turn_angle_deg=turn_angle_deg,
        )
        metrics.update(eval_metrics)
    return (agent, metrics) if return_agent else metrics
