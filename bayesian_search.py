"""Bayesian hyperparameter search for the DQN and PPO agents.

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to search the
hyperparameter space of either agent, reusing the existing training pipelines:

    - DQN : `train_dqn.train_DQN`
    - PPO : `train_ppo.run_ppo`

Both agents are scored with the SAME objective: the mean greedy success rate
from the shared `evaluation.evaluate_agent`, using one unified evaluation
config (same episodes, step cap, sigma, reward, dynamics and start). This makes
the DQN and PPO results directly comparable. Each objective is averaged over
several training seeds to reduce RL seed variance, so the search does not chase
a lucky run. After the search, re-run the best config over *more* seeds for the
final, reportable numbers.

Configure the run by editing the CONFIG block below, then simply:

    python bayesian_search.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna


# =========================================================================== #
# CONFIG  --  edit these instead of passing command-line arguments
# =========================================================================== #
ALGO = "dqn"                                # "dqn" or "ppo"
GRID = Path("grid_configs/restaurant_test2.npy")     # grid to search on
TRIALS = 30                                 # number of Optuna trials
TIMEOUT = None                              # wall-clock limit in seconds, or None
SEEDS = [0]                                 # training seeds averaged per trial (more = stabler, slower)
SAMPLER_SEED = 0                            # seed for the TPE sampler (reproducible search)
DEVICE = "cpu"                             # "cuda", "cuda:0", or "cpu"
SIGMA = 0.1                                 # training stochasticity
START_POS = None                            # (col, row) fixed start; None = grid start cell; "random" = random starts
RESULTS_DIR = Path("results")

# --- DQN training budget ---
DQN_BUDGET = 50_000                         # total training steps per trial
DQN_MAX_STEPS_PER_EPISODE = 200
DQN_EPSILON_DECAY_EPISODES = 150
DQN_TRAIN_START_MODE = "random"             # "fixed" or "random" (random = start curriculum)

# --- PPO training budget ---
PPO_EPISODES = 600                          # training episodes per trial
PPO_ITERS = 200                             # max steps per episode
PPO_REWARD = "high"                         # training reward fn: "default" or "high"
PPO_TRAIN_START_MODE = "random"              # "fixed" or "random"

# --- Unified evaluation (IDENTICAL for DQN and PPO) ---
# NOTE: this is the SEARCH's internal eval used to rank hyperparameters only.
# The final DQN-vs-PPO comparison (train_and_evaluate.py) is separate and uses
# a single fixed start. "random" eval start -> each of EVAL_EPISODES episodes
# uses a different (seeded, reproducible) random empty cell, so success becomes
# a continuous rate that matches random-start training and has low variance.
EVAL_START_MODE = "random"                 # "random" (multi-start) or "fixed"
EVAL_EPISODES = 10                          # eval episodes; with "random" = number of distinct start cells
EVAL_MAX_STEPS = 200                        # eval step cap
EVAL_SIGMA = 0.0                            # eval stochasticity
EVAL_REWARD = "default"                     # reward fn used to evaluate BOTH agents
EVAL_SEED = 1000                            # base eval seed (episode ep uses EVAL_SEED + ep)
AGENT_RADIUS = 0.5                          # env dynamics, shared by training and eval
MOVE_DISTANCE = 0.2
TURN_ANGLE_DEG = 15.0

# --- Objective + SPL (closeness to the approximate optimal path) ---
OBJECTIVE_METRIC = "goal_progress"                    # "success_rate", "spl", "train_sr_last100", or "goal_progress"
COMPUTE_SPL = True                          # compute SPL vs. lattice-BFS optimal (recorded either way)
SPL_POS_RES = 0.1                           # BFS position resolution for the optimal planner

# --- Final re-evaluation (statistical significance) ---
# After the search, the top-K configurations are retrained over a larger set of
# seeds and ranked by the MEAN objective (mean ± std reported), so the final
# pick is robust to seed luck rather than chosen from a single noisy trial.
FINAL_TOP_K = 5                             # how many best trials to re-evaluate (0 = skip)
FINAL_SEEDS = [0, 1, 2, 3, 4]              # training seeds for the re-evaluation
# =========================================================================== #


def resolve_device(device: str) -> str:
    """Validates the requested device; aborts early if CUDA was asked for but
    is unavailable, and prints the GPU that will be used."""
    if device.startswith("cuda"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SystemExit("PyTorch is required for CUDA. Install it first.") from exc
        if not torch.cuda.is_available():
            raise SystemExit(
                f"Requested DEVICE='{device}', but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build, or set DEVICE='cpu' in the CONFIG block."
            )
        idx = torch.cuda.current_device()
        print(f"CUDA available: using device '{device}' -> "
              f"{torch.cuda.get_device_name(idx)} (count={torch.cuda.device_count()})")
    else:
        print(f"Using device '{device}'.")
    return device


# --------------------------------------------------------------------------- #
# Shared evaluation (same procedure + metric for both agents)
# --------------------------------------------------------------------------- #
def evaluate(agent, cfg) -> dict:
    """Greedy evaluation of any trained agent via the shared evaluator."""
    from evaluation import evaluate_agent

    return evaluate_agent(
        agent,
        cfg["grid"],
        episodes=cfg["eval_episodes"],
        max_steps=cfg["eval_max_steps"],
        sigma=cfg["eval_sigma"],
        agent_start_pos=cfg["eval_start_pos"],
        seed=cfg["eval_seed"],
        reward_fn=cfg["eval_reward_fn"],
        agent_radius=cfg["agent_radius"],
        move_distance=cfg["move_distance"],
        turn_angle_deg=cfg["turn_angle_deg"],
        compute_spl=cfg["compute_spl"],
        spl_pos_res=cfg["spl_pos_res"],
    )


def _score(res, cfg, train_sr_last100=None) -> float:
    """Selects the configured objective metric.

    "success_rate"/"spl" come from the greedy eval `res`; "train_sr_last100" is
    the training success rate over the last 100 episodes (a denser signal that
    is non-zero before the greedy policy can fully solve the task, so TPE has a
    gradient to climb during the search).
    """
    metric = cfg["objective_metric"]
    if metric == "train_sr_last100":
        return float(train_sr_last100 or 0.0)
    key = {
        "success_rate": "eval_success_rate",
        "spl": "eval_avg_spl",
        "goal_progress": "eval_avg_goal_progress",
    }[metric]
    val = res.get(key)
    return 0.0 if val is None else float(val)


def _record(trial, seed, res):
    """Stores per-seed eval metrics on the trial for later inspection."""
    trial.set_user_attr(f"seed{seed}_success_rate", res["eval_success_rate"])
    trial.set_user_attr(f"seed{seed}_spl", res["eval_avg_spl"])
    trial.set_user_attr(f"seed{seed}_avg_steps", res["eval_avg_steps"])
    trial.set_user_attr(f"seed{seed}_avg_optimal", res.get("eval_avg_optimal_steps"))
    trial.set_user_attr(f"seed{seed}_goal_progress", res.get("eval_avg_goal_progress"))


def _aggregate(trial, sr_list, spl_list, train_sr_list=None):
    """Stores the mean success rate, mean SPL (and mean train SR) over seeds."""
    mean_sr = float(np.mean(sr_list)) if sr_list else 0.0
    spl_vals = [s for s in spl_list if s is not None]
    mean_spl = float(np.mean(spl_vals)) if spl_vals else 0.0
    trial.set_user_attr("mean_success_rate", mean_sr)
    trial.set_user_attr("mean_spl", mean_spl)
    tsr = [s for s in (train_sr_list or []) if s is not None]
    if tsr:
        trial.set_user_attr("mean_train_sr_last100", float(np.mean(tsr)))


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def suggest_dqn(trial) -> dict:
    return dict(
        reward=trial.suggest_categorical("reward", ["default", "high"]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.97, 0.99, 0.995]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        target_update_frequency=trial.suggest_categorical(
            "target_update_frequency", [250, 500, 1000, 2000]),
        replay_buffer_size=trial.suggest_categorical(
            "replay_buffer_size", [10_000, 50_000, 100_000]),
        epsilon_end=trial.suggest_categorical("epsilon_end", [0.0, 0.01, 0.05]),
        hidden_width=trial.suggest_categorical("hidden_width", [64, 128, 256]),
        hidden_depth=trial.suggest_int("hidden_depth", 2, 3),
    )


def _dqn_hidden_sizes(params: dict) -> tuple[int, ...]:
    """Builds the DQN network widths from searched hidden_width/hidden_depth."""
    return tuple([params["hidden_width"]] * params["hidden_depth"])


def _ppo_kwargs(raw: dict) -> dict:
    """Converts raw searched values into run_ppo kwargs.

    ``hidden_width`` + ``hidden_depth`` become ``hidden_sizes``. Used by both
    the sampler and the final re-evaluation (which reconstructs kwargs from a
    stored trial's params).
    """
    d = dict(raw)
    width = d.pop("hidden_width")
    depth = d.pop("hidden_depth")
    d["hidden_sizes"] = tuple([width] * depth)
    return d


def suggest_ppo(trial) -> dict:
    raw = dict(
        reward=trial.suggest_categorical("reward", ["default", "high"]),
        hidden_width=trial.suggest_categorical("hidden_width", [64, 128, 256]),
        hidden_depth=trial.suggest_int("hidden_depth", 1, 3),
        policy_lr=trial.suggest_float("policy_lr", 1e-5, 1e-3, log=True),
        value_lr=trial.suggest_float("value_lr", 1e-5, 3e-3, log=True),
        clip_epsilon=trial.suggest_float("clip_epsilon", 0.1, 0.3),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98]),
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]),
        entropy_coef=trial.suggest_float("entropy_coef", 1e-4, 5e-2, log=True),
        update_epochs=trial.suggest_categorical("update_epochs", [3, 4, 6, 10]),
        minibatch_size=trial.suggest_categorical("minibatch_size", [32, 64, 128, 256]),
        rollout_steps=trial.suggest_categorical("rollout_steps", [1024, 2048, 4096]),
        activation=trial.suggest_categorical("activation", ["tanh", "relu"]),
        # 1e6 effectively disables gradient clipping.
        max_grad_norm=trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0, 1e6]),
    )
    return _ppo_kwargs(raw)


def _eval_config(algo: str, params: dict, seed: int, cfg: dict) -> dict:
    """Trains one agent from `params` at `seed` and returns its eval result.

    `params` are the raw searched values (Optuna `trial.params`): for DQN they
    map straight onto `train_DQN`; for PPO they are converted via `_ppo_kwargs`.
    """
    if algo == "dqn":
        from train_dqn import train_DQN
        budget = cfg["budget"]
        short, mid = max(1, budget // 3), max(2, (2 * budget) // 3)
        agent, *_ = train_DQN(
            grid=cfg["grid"],
            short_train_steps_eval=short,
            mid_train_steps_eval=mid,
            max_steps_total=budget,
            n_episodes_epsilon_decay=cfg["epsilon_decay_episodes"],
            max_steps_per_episode=cfg["max_steps_per_episode"],
            sigma=cfg["sigma"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=1.0,
            epsilon_end=params["epsilon_end"],
            batch_size=params["batch_size"],
            replay_buffer_size=params["replay_buffer_size"],
            target_update_frequency=params["target_update_frequency"],
            random_seed=seed,
            agent_start_pos=cfg["start_pos"],
            no_gui=True,
            device=cfg["device"],
            reward=params["reward"],
            agent_radius=cfg["agent_radius"],
            move_distance=cfg["move_distance"],
            turn_angle_deg=cfg["turn_angle_deg"],
            hidden_sizes=_dqn_hidden_sizes(params),
            train_start_mode=cfg["dqn_train_start_mode"],
        )
        return evaluate(agent, cfg)

    from train_ppo import run_ppo
    kwargs = _ppo_kwargs(params)
    agent, _ = run_ppo(
        cfg["grid"],
        sigma=cfg["sigma"],
        seed=seed,
        device=cfg["device"],
        episodes=cfg["episodes"],
        iters=cfg["iters"],
        start_pos=cfg["start_pos"],
        train_start_mode=cfg["train_start_mode"],
        agent_radius=cfg["agent_radius"],
        move_distance=cfg["move_distance"],
        turn_angle_deg=cfg["turn_angle_deg"],
        do_eval=False,
        return_agent=True,
        **kwargs,
    )
    return evaluate(agent, cfg)


# --------------------------------------------------------------------------- #
# Objectives  (same metric: mean greedy eval success rate)
# --------------------------------------------------------------------------- #
def objective_dqn(trial, cfg) -> float:
    from train_dqn import train_DQN

    params = suggest_dqn(trial)
    budget = cfg["budget"]
    short, mid = max(1, budget // 3), max(2, (2 * budget) // 3)

    scores, sr_list, spl_list, train_sr_list = [], [], [], []
    for seed in cfg["seeds"]:
        agent, history, short_agent, mid_agent = train_DQN(
            grid=cfg["grid"],
            short_train_steps_eval=short,
            mid_train_steps_eval=mid,
            max_steps_total=budget,
            n_episodes_epsilon_decay=cfg["epsilon_decay_episodes"],
            max_steps_per_episode=cfg["max_steps_per_episode"],
            sigma=cfg["sigma"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=1.0,
            epsilon_end=params["epsilon_end"],
            batch_size=params["batch_size"],
            replay_buffer_size=params["replay_buffer_size"],
            target_update_frequency=params["target_update_frequency"],
            random_seed=seed,
            agent_start_pos=cfg["start_pos"],
            no_gui=True,
            device=cfg["device"],
            reward=params["reward"],
            agent_radius=cfg["agent_radius"],
            move_distance=cfg["move_distance"],
            turn_angle_deg=cfg["turn_angle_deg"],
            hidden_sizes=_dqn_hidden_sizes(params),
            train_start_mode=cfg["dqn_train_start_mode"],
        )
        # Training success rate over the last 100 episodes.
        last100 = history[-100:]
        train_sr = float(np.mean([ep["terminated"] for ep in last100])) if last100 else 0.0
        res = evaluate(agent, cfg)
        scores.append(_score(res, cfg, train_sr))
        sr_list.append(res["eval_success_rate"])
        spl_list.append(res["eval_avg_spl"])
        train_sr_list.append(train_sr)
        _record(trial, seed, res)
        trial.set_user_attr(f"seed{seed}_train_sr_last100", train_sr)
        # Record sample-efficiency checkpoints (same evaluator) for inspection.
        trial.set_user_attr(f"seed{seed}_short_sr", evaluate(short_agent, cfg)["eval_success_rate"])
        trial.set_user_attr(f"seed{seed}_mid_sr", evaluate(mid_agent, cfg)["eval_success_rate"])

    _aggregate(trial, sr_list, spl_list, train_sr_list)
    return float(np.mean(scores))


def objective_ppo(trial, cfg) -> float:
    from train_ppo import run_ppo

    params = suggest_ppo(trial)
    scores, sr_list, spl_list, train_sr_list = [], [], [], []
    for seed in cfg["seeds"]:
        # `reward` is a searched hyperparameter (inside params), so it is not
        # passed separately here.
        agent, train_metrics = run_ppo(
            cfg["grid"],
            sigma=cfg["sigma"],
            seed=seed,
            device=cfg["device"],
            episodes=cfg["episodes"],
            iters=cfg["iters"],
            start_pos=cfg["start_pos"],
            train_start_mode=cfg["train_start_mode"],
            agent_radius=cfg["agent_radius"],
            move_distance=cfg["move_distance"],
            turn_angle_deg=cfg["turn_angle_deg"],
            do_eval=False,            # objective uses the shared evaluator below
            return_agent=True,
            **params,
        )
        train_sr = train_metrics.get("train_success_rate_last_100")
        res = evaluate(agent, cfg)
        scores.append(_score(res, cfg, train_sr))
        sr_list.append(res["eval_success_rate"])
        spl_list.append(res["eval_avg_spl"])
        train_sr_list.append(train_sr)
        _record(trial, seed, res)
        trial.set_user_attr(f"seed{seed}_train_sr_last100", train_sr)

    _aggregate(trial, sr_list, spl_list, train_sr_list)
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Study driver
# --------------------------------------------------------------------------- #
def run_search(algo: str, cfg: dict, n_trials: int, sampler_seed: int,
               timeout: float | None = None) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{algo}_bayes",
    )
    objective = {"dqn": objective_dqn, "ppo": objective_ppo}[algo]

    def _log(study_, trial_):
        best = study_.best_trial
        sr = trial_.user_attrs.get("mean_success_rate")
        spl = trial_.user_attrs.get("mean_spl")
        metrics = ""
        if sr is not None and spl is not None:
            metrics = f"  success={sr:.3f} SPL={spl:.3f}"
        print(f"[{algo}] trial {trial_.number:>3}  value={trial_.value:.4f}{metrics}  "
              f"| best={best.value:.4f} (trial {best.number})")

    study.optimize(lambda t: objective(t, cfg), n_trials=n_trials,
                   timeout=timeout, callbacks=[_log])
    return study


def save_study(study: optuna.Study, algo: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    fp = out_dir / f"bayes_{algo}_{ts}.json"
    payload = {
        "algo": algo,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    fp.write_text(json.dumps(payload, indent=2))
    return fp


# --------------------------------------------------------------------------- #
# Final re-evaluation of the top-K configs over more seeds
# --------------------------------------------------------------------------- #
def final_reeval(study, algo, cfg, top_k, seeds):
    """Retrains the top-K trial configs over `seeds` and ranks by mean objective.

    Returns a list of result dicts (sorted best-first) with mean +/- std of
    success rate and SPL across the seeds, so the final pick is robust to the
    seed noise of any single search trial.
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    top = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]
    if not top:
        print("\nNo completed trials to re-evaluate.")
        return []

    print(f"\n=== Final re-evaluation: top {len(top)} configs over seeds {seeds} ===")
    results = []
    for rank, t in enumerate(top, 1):
        sr_list, spl_list = [], []
        for seed in seeds:
            res = _eval_config(algo, t.params, seed, cfg)
            sr_list.append(res["eval_success_rate"])
            if res["eval_avg_spl"] is not None:
                spl_list.append(res["eval_avg_spl"])
        sr_mean, sr_std = float(np.mean(sr_list)), float(np.std(sr_list))
        spl_mean = float(np.mean(spl_list)) if spl_list else float("nan")
        spl_std = float(np.std(spl_list)) if spl_list else float("nan")
        results.append({
            "trial": t.number,
            "search_value": t.value,
            "params": t.params,
            "success_rate_mean": sr_mean,
            "success_rate_std": sr_std,
            "spl_mean": spl_mean,
            "spl_std": spl_std,
            "success_rate_per_seed": sr_list,
            "spl_per_seed": spl_list,
            "seeds": list(seeds),
        })
        print(f"  [{rank}] trial {t.number:>3}: "
              f"success={sr_mean:.3f}±{sr_std:.3f}  SPL={spl_mean:.3f}±{spl_std:.3f}")

    # Rank by the mean of the chosen objective metric (robust to seed luck).
    key = "spl_mean" if cfg["objective_metric"] == "spl" else "success_rate_mean"
    results.sort(key=lambda e: (e[key] if e[key] == e[key] else -1.0), reverse=True)
    return results


def save_final(results, algo, out_dir) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    fp = out_dir / f"bayes_{algo}_{ts}_final.json"
    fp.write_text(json.dumps(results, indent=2))
    return fp


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _resolve_start_pos():
    if START_POS == "random":
        return None
    if START_POS is None:
        from train_ppo import parse_start_pos
        return parse_start_pos(None, GRID)
    return tuple(START_POS)


def main():
    device = resolve_device(DEVICE)
    start_pos = _resolve_start_pos()

    from world.environment_continuous import EnvironmentContinuous
    eval_reward_fn = {
        "default": EnvironmentContinuous._default_reward_function,
        "high": EnvironmentContinuous._high_reward_function,
    }[EVAL_REWARD]

    # Eval start: "random" -> None so the env samples a (seeded, reproducible)
    # random empty cell each episode; "fixed" -> the training start cell.
    eval_start_pos = None if EVAL_START_MODE == "random" else start_pos

    cfg = {
        "grid": GRID,
        "seeds": SEEDS,
        "device": device,
        "sigma": SIGMA,
        "start_pos": start_pos,
        "eval_start_pos": eval_start_pos,
        # DQN training
        "budget": DQN_BUDGET,
        "max_steps_per_episode": DQN_MAX_STEPS_PER_EPISODE,
        "epsilon_decay_episodes": DQN_EPSILON_DECAY_EPISODES,
        "dqn_train_start_mode": DQN_TRAIN_START_MODE,
        # PPO training
        "episodes": PPO_EPISODES,
        "iters": PPO_ITERS,
        "reward": PPO_REWARD,
        "train_start_mode": PPO_TRAIN_START_MODE,
        # Unified evaluation (identical for both agents)
        "eval_episodes": EVAL_EPISODES,
        "eval_max_steps": EVAL_MAX_STEPS,
        "eval_sigma": EVAL_SIGMA,
        "eval_seed": EVAL_SEED,
        "eval_reward_fn": eval_reward_fn,
        "agent_radius": AGENT_RADIUS,
        "move_distance": MOVE_DISTANCE,
        "turn_angle_deg": TURN_ANGLE_DEG,
        "objective_metric": OBJECTIVE_METRIC,
        "compute_spl": COMPUTE_SPL,
        "spl_pos_res": SPL_POS_RES,
    }

    print(f"Bayesian (TPE) search: algo={ALGO}, trials={TRIALS}, "
          f"seeds/trial={SEEDS}, grid={GRID}, start={start_pos}")
    print(f"Unified eval: {EVAL_EPISODES} eps ({EVAL_START_MODE} start), "
          f"max_steps={EVAL_MAX_STEPS}, sigma={EVAL_SIGMA}, reward={EVAL_REWARD}")
    study = run_search(ALGO, cfg, TRIALS, SAMPLER_SEED, TIMEOUT)

    print("\n=== Best configuration ===")
    print(f"objective (mean eval {OBJECTIVE_METRIC}) = {study.best_value:.4f}")
    best_attrs = study.best_trial.user_attrs
    print(f"  success rate = {best_attrs.get('mean_success_rate', float('nan')):.4f}")
    print(f"  SPL          = {best_attrs.get('mean_spl', float('nan')):.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_fp = save_study(study, ALGO, RESULTS_DIR)
    print(f"\nStudy saved to {out_fp}")

    # Robust final selection: retrain the top-K configs over more seeds and
    # rank by the mean objective (mean ± std), so the winner isn't seed luck.
    if FINAL_TOP_K > 0 and FINAL_SEEDS:
        final = final_reeval(study, ALGO, cfg, FINAL_TOP_K, FINAL_SEEDS)
        if final:
            best = final[0]
            print(f"\n=== Final winner (by mean {OBJECTIVE_METRIC} over {len(FINAL_SEEDS)} seeds) ===")
            print(f"  trial {best['trial']}: "
                  f"success={best['success_rate_mean']:.3f}±{best['success_rate_std']:.3f}  "
                  f"SPL={best['spl_mean']:.3f}±{best['spl_std']:.3f}")
            for k, v in best["params"].items():
                print(f"    {k}: {v}")
            final_fp = save_final(final, ALGO, RESULTS_DIR)
            print(f"Final re-evaluation saved to {final_fp}")


if __name__ == "__main__":
    main()
