"""Unified agent evaluation.

A single, agent-agnostic evaluation routine so DQN and PPO (or any
`BaseAgent`) are judged with *exactly* the same procedure and metrics -- a
prerequisite for a fair comparison.

The agent only needs `take_action(state)`. Greedy/eval mode and per-episode
resets are handled generically:

    - greedy mode : `set_training(False)` (PPO) and/or `epsilon = 0` (DQN),
                    restored afterwards.
    - per episode : `new_episode(state)` (PPO) or `reset_episode()` (DQN) if present.

Every episode rebuilds the environment with `random_seed = seed + ep` so the
randomised start heading varies across episodes (giving a meaningful success
distribution even with sigma = 0 and a fixed start cell).
"""

from __future__ import annotations

import random
from datetime import datetime
from math import radians
from pathlib import Path
from tqdm import trange

import numpy as np
from shapely.geometry import Point

from world.environment_continuous import EnvironmentContinuous

# Cache of approximate optimal step counts so the (deterministic) lattice BFS is
# run at most once per start cell + dynamics, even across many evaluate_agent
# calls (e.g. every trial of a hyperparameter search).
_OPTIMAL_CACHE: dict = {}


def _goal_distance(env, polys) -> float:
    """Euclidean distance from the agent centre to the nearest target region."""
    if not polys:
        return 0.0
    p = Point(env.x, env.y)
    return min(poly.distance(p) for poly in polys)


# --------------------------------------------------------------------------- #
# Agent mode handling (works for both DQN and PPO)
# --------------------------------------------------------------------------- #
def _enter_eval_mode(agent) -> dict:
    saved: dict = {}
    if hasattr(agent, "set_training"):
        saved["train_mode"] = getattr(agent, "train_mode", None)
        agent.set_training(False)
    if hasattr(agent, "epsilon"):
        saved["epsilon"] = agent.epsilon
        agent.epsilon = 0.0
    return saved


def _exit_eval_mode(agent, saved: dict):
    if "epsilon" in saved:
        agent.epsilon = saved["epsilon"]
    if saved.get("train_mode") is not None:
        agent.set_training(saved["train_mode"])


def _begin_episode(agent, state):
    if hasattr(agent, "new_episode"):
        agent.new_episode(state)
    elif hasattr(agent, "reset_episode"):
        agent.reset_episode()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_agent(
    agent,
    grid_fp,
    *,
    episodes: int = 1,
    max_steps: int = 1000,
    sigma: float = 0.0,
    agent_start_pos=None,
    seed: int = 0,
    reward_fn=None,
    agent_radius: float = 0.2,
    move_distance: float = 0.2,
    turn_angle_deg: float = 15.0,
    optimal_steps: int | None = None,
    compute_spl: bool = False,
    spl_pos_res: float = 0.1,
    spl_free_initial_heading: bool = True,
    success_ratio_threshold: float = 1.2,
    save_image: bool = False,
) -> dict:
    """Runs a greedy evaluation of `agent` and returns standardized metrics.

    Args:
        agent: Any agent exposing `take_action(state)`.
        grid_fp: Grid file path.
        episodes: Number of evaluation episodes.
        max_steps: Step cap per episode.
        sigma: Environment stochasticity during evaluation.
        agent_start_pos: Fixed start cell (col, row), or None for random.
        seed: Base random seed; episode ep uses seed + ep.
        reward_fn: Reward function. Defaults to the env's default reward.
        agent_radius, move_distance, turn_angle_deg: Environment dynamics.
        optimal_steps: Fixed optimal-step count for SPL. Overridden per episode
            when `compute_spl` is True.
        compute_spl: If True, the approximate optimal number of actions from
            each episode's start cell is computed with a state-lattice BFS
            (`optimal_path.approx_optimal_steps`) and used for a proper SPL =
            success * optimal / max(optimal, steps). Results are cached per
            start cell.
        spl_pos_res: Position resolution (m) for the BFS planner.
        spl_free_initial_heading: Treat the initial orientation as free in the
            planner (the agent's real start heading is random).
        save_image: If True, saves a trajectory image of the last episode.

    Returns:
        Dict of evaluation metrics (identical schema for every agent type).
    """
    grid_fp = Path(grid_fp)
    if reward_fn is None:
        reward_fn = EnvironmentContinuous._default_reward_function

    saved = _enter_eval_mode(agent)

    successes, steps_list, rewards, spls, optimals = [], [], [], [], []
    failed_moves_list = []           # collisions (failed moves) per episode
    step_ratios = []                 # actual / optimal, successful episodes only
    step_ratio_per_episode = []      # aligned with episodes (None where N/A)
    goal_progress_list = []          # 1 - min_dist/initial_dist (closest approach)
    last_env = last_path = last_actions = None

    # Build the environment once (geometry is cached across resets); vary the
    # episode RNG by re-seeding the global `random` stream before each reset,
    # which reproduces what a per-episode `random_seed = seed + ep` would give.
    env = EnvironmentContinuous(
        grid_fp=grid_fp,
        no_gui=True,
        sigma=sigma,
        agent_start_pos=agent_start_pos,
        random_seed=seed,
        reward_fn=reward_fn,
        target_fps=-1,
        agent_radius=agent_radius,
        move_distance=move_distance,
        turn_angle=radians(turn_angle_deg),
    )

    try:
        for ep in trange(episodes):
            random.seed(seed + ep)
            state = env.reset()
            _begin_episode(agent, state)

            # Per-episode approximate optimal (lattice BFS), cached per start cell.
            ep_optimal = optimal_steps
            if compute_spl:
                cell = (int(env.x), int(env.y))
                cache_key = (str(grid_fp), cell, round(agent_radius, 4),
                             round(move_distance, 4), round(turn_angle_deg, 4),
                             round(spl_pos_res, 4), spl_free_initial_heading)
                if cache_key not in _OPTIMAL_CACHE:
                    from optimal_path import approx_optimal_steps
                    _OPTIMAL_CACHE[cache_key] = approx_optimal_steps(
                        env, (env.x, env.y),
                        pos_res=spl_pos_res,
                        free_initial_heading=spl_free_initial_heading,
                    )
                ep_optimal = _OPTIMAL_CACHE[cache_key]

            # Closest-approach progress toward the goal (dense, always computed).
            goal_polys = list(env.targets)
            initial_goal_dist = _goal_distance(env, goal_polys)
            min_goal_dist = initial_goal_dist

            path = [(env.x, env.y)]
            actions = []
            total_reward = 0.0
            n_steps = 0
            reached = False

            for _ in range(max_steps):
                action = agent.take_action(state)
                state, reward, terminated, _info = env.step(action)
                actions.append(int(action))
                total_reward += reward
                n_steps += 1
                path.append((env.x, env.y))
                min_goal_dist = min(min_goal_dist, _goal_distance(env, goal_polys))
                if terminated:
                    reached = True
                    break

            # 1 - normalized distance to goal: 1 = reached, 0 = no progress made.
            if reached or initial_goal_dist <= 1e-9:
                progress = 1.0
            else:
                progress = max(0.0, 1.0 - min_goal_dist / initial_goal_dist)
            goal_progress_list.append(progress)

            successes.append(1.0 if reached else 0.0)
            steps_list.append(n_steps)
            rewards.append(total_reward)
            failed_moves_list.append(env.world_stats.get("total_failed_moves", 0))
            ratio = None
            if ep_optimal:
                optimals.append(ep_optimal)
                spls.append((ep_optimal / max(ep_optimal, n_steps)) if reached else 0.0)
                # Step ratio: actual / optimal (1.0 = optimal, 1.01 = 1% over).
                # Only meaningful when the agent actually reached the target.
                if reached:
                    ratio = n_steps / ep_optimal
                    step_ratios.append(ratio)
            step_ratio_per_episode.append(ratio)

            last_env, last_path, last_actions = env, path, actions
    finally:
        _exit_eval_mode(agent, saved)

    n_success = int(sum(successes))
    n_eps = len(step_ratio_per_episode)
    # Path-quality rates (need an optimal reference, i.e. compute_spl or
    # optimal_steps). within_threshold_rate: reached AND steps <= threshold x
    # optimal (default 1.2 = within 20%). optimal_rate: reached AND steps <=
    # optimal. None when no optimal was available.
    if optimals:
        within_rate = sum(1 for r in step_ratio_per_episode
                          if r is not None and r <= success_ratio_threshold) / n_eps
        optimal_rate = sum(1 for r in step_ratio_per_episode
                           if r is not None and r <= 1.0) / n_eps
    else:
        within_rate = optimal_rate = None
    result = {
        "eval_episodes": int(episodes),
        "eval_successes": n_success,
        "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval_within_threshold_rate": within_rate,
        "eval_optimal_rate": optimal_rate,
        "eval_success_ratio_threshold": success_ratio_threshold,
        "eval_avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "eval_avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "eval_total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "eval_avg_failed_moves": float(np.mean(failed_moves_list)) if failed_moves_list else 0.0,
        "eval_avg_spl": (float(np.mean(spls)) if spls else None),
        "eval_avg_optimal_steps": (float(np.mean(optimals)) if optimals else None),
        # Mean ratio of actual to optimal steps over *successful* episodes:
        # 1.0 = optimal, 1.01 = 1% more steps than the approximate optimal.
        "eval_avg_step_ratio": (float(np.mean(step_ratios)) if step_ratios else None),
        # Mean closest-approach progress to the goal: 1 - min_dist/initial_dist.
        "eval_avg_goal_progress": float(np.mean(goal_progress_list)) if goal_progress_list else 0.0,
        "eval_steps_per_episode": steps_list,
        "eval_success_per_episode": [int(s) for s in successes],
        "eval_reward_per_episode": rewards,
        "eval_failed_moves_per_episode": failed_moves_list,
        "eval_optimal_per_episode": optimals,
        "eval_step_ratio_per_episode": step_ratio_per_episode,
        "eval_goal_progress_per_episode": goal_progress_list,
        "eval_success_rate_ci": (
            (
                float(np.mean(successes) - 1.96 * np.std(successes, ddof=1) / np.sqrt(len(successes))),
                float(np.mean(successes) + 1.96 * np.std(successes, ddof=1) / np.sqrt(len(successes))),
            ) if len(successes) > 1 else None
        ),
        "eval_avg_spls_ci": (
            (
                float(np.mean(spls) - 1.96 * np.std(spls, ddof=1) / np.sqrt(len(spls))),
                float(np.mean(spls) + 1.96 * np.std(spls, ddof=1) / np.sqrt(len(spls))),
            ) if len(spls) > 1 else None
        ),
        "eval_avg_failed_moves_ci": (
            (
                float(np.mean(failed_moves_list) - 1.96 * np.std(failed_moves_list, ddof=1) / np.sqrt(len(failed_moves_list))),
                float(np.mean(failed_moves_list) + 1.96 * np.std(failed_moves_list, ddof=1) / np.sqrt(len(failed_moves_list))),
            ) if len(failed_moves_list) > 1 else None
        ),
    }
    for key, src in [
        ("spls", "eval_spls_per_episode"),
        ("successes", "eval_success_per_episode"),
        ("failed_moves", "eval_failed_moves_per_episode"),
    ]:
        values = np.asarray(result.get(src, []), dtype=float)

        if len(values) > 1:
            mean = np.mean(values)
            sem = np.std(values, ddof=1) / np.sqrt(len(values))
            result[f"{key}_95ci"] = (
                mean - 1.96 * sem,
                mean + 1.96 * sem,
            )
        elif len(values) == 1:
            result[f"{key}_95ci"] = (values[0], values[0])
        else:
            result[f"{key}_95ci"] = None

    if last_env is not None:
        result["eval_final_pos"] = [round(float(last_env.x), 4), round(float(last_env.y), 4)]
        result["eval_path"] = [
            [round(float(x), 4), round(float(y), 4), a]
            for (x, y), a in zip(last_path, last_actions + [None])
        ]
        result["eval_world_stats"] = dict(last_env.world_stats)

        if save_image:
            from world.helpers import save_results
            img = last_env.trajectory_image(last_path)
            file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            save_results(file_name, last_env.world_stats, img, show_images=False)
            print(f"\n>>> Results saved to results/{file_name}.png / .txt "
                  f"({len(last_path)} steps) <<<")

    return result
