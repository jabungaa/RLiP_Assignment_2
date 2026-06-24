"""
Training and evaluation pipeline by importing existing functionalities from PPO and DQN
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Pipeline imports. PPO training and evaluation both live in train_ppo.py now
# (step-budgeted train_ppo + evaluation.py-based evaluate_ppo), consistent with
# the DQN pipeline. train_ppo_functions is no longer used.
from train_dqn import train_DQN, evaluate_DQN
from train_ppo import train_ppo, evaluate_ppo, make_training_start_sampler

# PPO-specific imports needed for instantiation
from agents.PPO import PPO_agent
from world.environment_continuous import EnvironmentContinuous
from world.path_visualizer import visualize_path, save_path_image
import torch
import random
import os
from math import radians

def set_all_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)  # controls Python hash randomness

print("FILE LOADED")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DQN and/or PPO agents.")

    # Shared configurations
    parser.add_argument("--grid", type=Path, default=Path("grid_configs/restaurant_test.npy"), help="Path to the .npy grid file.")
    parser.add_argument("--agents", choices=("dqn", "ppo", "both"), default="both", help="Which agent(s) to run.")
    parser.add_argument("--ppo_sigma", type=float, default=0.05, help="Environment stochasticity (training).")
    parser.add_argument("--dqn_sigma", type=float, default=0.1, help="Environment stochasticity (training).")
    parser.add_argument("--eval_sigma", type=float, default=0.0, help="Environment stochasticity (evaluation).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_pos", type=str, default=None, help="Agent start position as row,col.")
    parser.add_argument("--no_gui", action="store_true", default=True,
                    help="Disable GUI during training.")
    parser.add_argument("--eval_gui", action="store_true", default=False, help="Enable GUI during evaluation.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes for evaluation.")
    parser.add_argument("--agent_radius", type=float, default=0.2, help="Agent radius for evaluation.")
    parser.add_argument("--move_distance", type=float, default=0.5, help="Agent move distance for evaluation.")
    parser.add_argument("--turn_angle_deg", type=float, default=15.0, help="Agent turn angle for evaluation.")

    # DQN Hyperparameters
    dqn = parser.add_argument_group("DQN")
    dqn.add_argument("--dqn_episodes", type=int, default=5)
    dqn.add_argument("--dqn_max_steps_total", type=int, default=100)
    dqn.add_argument("--dqn_short_train", type=int, default=25)
    dqn.add_argument("--dqn_mid_train", type=int, default=50)
    dqn.add_argument("--dqn_max_steps_per_episode", type=int, default=1)
    dqn.add_argument("--dqn_lr", type=float, default=0.001)
    dqn.add_argument("--dqn_gamma", type=float, default=0.99)

    # PPO Hyperparameters
    ppo = parser.add_argument_group("PPO")
    # ppo.add_argument("--ppo_episodes", type=int, default=5)
    ppo.add_argument("--ppo_max_steps_total", type=int, default=100)
    ppo.add_argument("--ppo_short_train", type=int, default=25)
    ppo.add_argument("--ppo_mid_train", type=int, default=50)
    ppo.add_argument("--ppo_max_steps_per_episode", type=int, default=1)
    ppo.add_argument("--ppo_eval_steps", type=int, default=200)
    ppo.add_argument("--ppo_policy_lr", type=float, default=3e-4)
    ppo.add_argument("--ppo_value_lr", type=float, default=1e-3)
    
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

def parse_start_pos(raw: str | None, grid_fp: Path) -> tuple[int, int] | None:
    if raw:
        row, col = raw.split(",")
        return int(row), int(col)
    if grid_fp.exists():
        grid = np.load(grid_fp)
        starts = np.argwhere(grid == 4)
        if len(starts) > 0: return int(starts[0][0]), int(starts[0][1])
        empty = np.argwhere(grid == 0)
        if len(empty) > 0: return int(empty[0][0]), int(empty[0][1])
    return (1, 1)

def print_comparison(results: dict):
    print(f"\n{'-'*60}\n  Summary - Training & Evaluation\n{'-'*60}")
    agents = list(results.keys())
    print(f"  {'Metric':<25}" + "".join(f"{a.upper():>15}" for a in agents))
    print("  " + "-" * (25 + 15 * len(agents)))
    
    metrics = [
        ("train_success_rate", "Train Success Rate"),
        ("eval_success_rate", "Eval Success (<=1.2 opt)"),
        ("optimal_rate", "Optimal Rate"),
        ("eval_total_reward", "Eval Reward"),
        ("eval_steps", "Eval Steps"),
        ("eval_spl", "Eval SPL"),
        ("eval_avg_collisions", "Eval Avg. Collisions"),
        ("short_train_eval_spl", "Short Train Eval SPL"),
        ("mid_train_eval_spl", "Mid Train Eval SPL"),
        ("auc", "AUC (SPL vs Iterations)"),
    ]
    for key, label in metrics:
        vals = [f"{results[a].get(key, '—'):.3f}" if isinstance(results[a].get(key), float) else str(results[a].get(key, '—')) for a in agents]
        print(f"  {label:<25}" + "".join(f"{v:>15}" for v in vals))
    print()

def training_convergence_plot(history: list[dict], agent_name: str, 
                               results_dir: Path, stamp: str, checking_window: int = 50):
    """ Plots the steps per episode and rolling avg episode length every 50 training episodes to show convergence"""
    steps_per_ep = [ep["steps"] for ep in history]
    steps = np.cumsum([ep["steps"] for ep in history])
    
    #use rolling average for smoothing
    rolling_avg = np.convolve(steps_per_ep, np.ones(checking_window)/checking_window, mode='valid')
    rolling_avg_steps = steps[checking_window-1:]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, steps_per_ep, alpha=0.2, color="orange", label="Raw")
    ax.plot(rolling_avg_steps, rolling_avg, color="orange", 
            label=f"Rolling Avg. (window={checking_window})")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Steps per Episode")
    ax.set_title(f"{agent_name} — Steps per Episode in Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = results_dir / f"{stamp}_{agent_name}_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved to {out_path}")
    return out_path

def path_quality_rates(agent, grid, start_pos, sigma, seed, episodes, max_steps,
                       within=1.2, agent_radius=0.5, move_distance=0.2,
                       turn_angle_deg=15.0):
    """Thin wrapper that asks the shared evaluator for the path-quality rates.

    success_rate = reach within `within` x the approximate optimal path length
    (default 1.2 = 20%); optimal_rate = reach in <= optimal. The thresholding
    itself lives in `evaluation.evaluate_agent`.
    """
    from evaluation import evaluate_agent
    res = evaluate_agent(
        agent, grid, episodes=episodes, max_steps=max_steps, sigma=sigma,
        agent_start_pos=start_pos, seed=seed, agent_radius=agent_radius,
        move_distance=move_distance, turn_angle_deg=turn_angle_deg,
        compute_spl=True, success_ratio_threshold=within,
    )
    return (res["eval_within_threshold_rate"] or 0.0,
            res["eval_optimal_rate"] or 0.0)


def main():
    print("ENTERED MAIN")
    args = parse_args()
    set_all_seeds(args.seed)

    
    
    stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    start_pos = parse_start_pos(args.start_pos, args.grid)

    # Approximate optimal step count from the (fixed) comparison start, used as
    # the SPL baseline for BOTH agents so their SPL/AUC are directly comparable.
    from optimal_path import approx_optimal_for_grid
    baseline_steps = approx_optimal_for_grid(
        args.grid, start_pos, agent_radius=0.5, move_distance=0.2, turn_angle_deg=15.0)
    print(f"SPL baseline (approx optimal from {start_pos}): {baseline_steps} steps")

    all_results = {}
    print("agents =", args.agents)
    
    print("device =", args.device)

    # ── DQN Pipeline ────
    if args.agents in ("dqn", "both"):
        print(f"\nStarting DQN Pipeline----------------------------------------------------------------------")
        
        # train DQN
        dqn_agent, dqn_history, short_train_agent, mid_train_agent = train_DQN(
            grid=args.grid,
            short_train_steps_eval= args.dqn_short_train,
            mid_train_steps_eval= args.dqn_mid_train,
            max_steps_total= args.dqn_max_steps_total,
            n_episodes_epsilon_decay=args.dqn_episodes,
            max_steps_per_episode=args.dqn_max_steps_per_episode,
            sigma=args.dqn_sigma,
            learning_rate=args.dqn_lr,
            gamma=args.dqn_gamma,
            random_seed=args.seed,
            agent_start_pos=start_pos,
            no_gui=args.no_gui,
            device=args.device,
            # Same dynamics as PPO for a fair head-to-head comparison.
            agent_radius=args.agent_radius,
            move_distance=args.move_distance,
            turn_angle_deg=args.turn_angle_deg,
        )

        training_convergence_plot(dqn_history, "DQN", args.results_dir, stamp, checking_window=50)
        
        dqn_train_successes = sum(1 for ep in dqn_history if ep["terminated"])
        dqn_metrics = {
            "train_total_successes": dqn_train_successes,
            "train_total_episodes": len(dqn_history),
            "train_success_rate": dqn_train_successes / len(dqn_history) if dqn_history else 0.0
        }

        
        # Evaluate DQN and gather results
        short_train_dqn_eval = evaluate_DQN(
            agent=short_train_agent,
            grid=args.grid,
            max_steps_per_episode=args.dqn_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            no_gui=True,
            episodes=args.eval_episodes,
            agent_radius=args.agent_radius,
            move_distance=args.move_distance,
            turn_angle_deg=args.turn_angle_deg,
            optimal_steps=baseline_steps,
        )

        mid_train_dqn_eval = evaluate_DQN(
            agent=mid_train_agent,
            grid=args.grid,
            max_steps_per_episode=args.dqn_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            no_gui=True,
            episodes=args.eval_episodes,
            agent_radius=args.agent_radius,
            move_distance=args.move_distance,
            turn_angle_deg=args.turn_angle_deg,
            optimal_steps=baseline_steps,
        )

        dqn_eval = evaluate_DQN(
            agent=dqn_agent,
            grid=args.grid,
            max_steps_per_episode=args.dqn_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            no_gui=True,
            episodes=args.eval_episodes,
            agent_radius=args.agent_radius,
            move_distance=args.move_distance,
            turn_angle_deg=args.turn_angle_deg,
            optimal_steps=baseline_steps,
        )
        
        # Map DQN keys to standardized metrics keys
        dqn_eval_metrics = {
            "eval_total_reward": dqn_eval.get("total_reward", 0.0),
            "eval_steps": dqn_eval.get("avg_steps", 0.0),
            "eval_success_rate": dqn_eval.get("eval_success_rate", 0.0),
            "eval_spl": dqn_eval.get("SPL", 0.0),
            "eval_avg_collisions":dqn_eval.get("avg_failed_moves", 0.0),
            "short_train_eval_spl": short_train_dqn_eval.get("SPL", 0.0),
            "mid_train_eval_spl": mid_train_dqn_eval.get("SPL", 0.0),
        }

        # Redefine success as reaching within 20% of the optimal path length,
        # and report the rate of achieving the optimal path length.
        dqn_success_rate, dqn_optimal_rate = path_quality_rates(
            dqn_agent, args.grid, start_pos, args.eval_sigma, args.seed,
            args.eval_episodes, args.dqn_max_steps_per_episode)
        dqn_eval_metrics["eval_success_rate"] = dqn_success_rate
        dqn_eval_metrics["optimal_rate"] = dqn_optimal_rate

        # Collect data points
        steps = [args.dqn_short_train, args.dqn_mid_train, args.dqn_max_steps_total]
        spls  = [
            dqn_eval_metrics["short_train_eval_spl"],
            dqn_eval_metrics["mid_train_eval_spl"],
            dqn_eval_metrics["eval_spl"],
        ]

        # Sort by steps (x-axis) in case they aren't already ordered
        steps, spls = zip(*sorted(zip(steps, spls)))
        steps = [0] + list(steps)
        spls  = [0] + list(spls)

        # Calculate AUC using the trapezoidal rule
        dqn_auc = trapezoid(spls, steps)

        all_results["dqn"] = {**dqn_metrics, **dqn_eval_metrics, "auc":dqn_auc,"agent": "DQN"}

        # save path image
        EnvironmentContinuous.evaluate_agent(
            grid_fp=args.grid,
            agent=dqn_agent,
            max_steps=args.dqn_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
            no_gui=args.no_gui
        )

    # ── PPO Pipeline ─────────────────────────────────────────────────────────
    if args.agents in ("ppo", "both"):
        print(f"\nStarting PPO Pipeline----------------------------------------------------------------------")

        # Dynamics MUST match evaluation (radius 0.2, move 0.5) — these are the
        # values trial 14 was tuned with in the Bayesian search.
        env = EnvironmentContinuous(
            grid_fp=args.grid,
            no_gui=args.no_gui,
            sigma=args.ppo_sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
            reward_fn=EnvironmentContinuous._default_reward_function,
            agent_radius=args.agent_radius,
            move_distance=args.move_distance,
            turn_angle=radians(args.turn_angle_deg),
        )

        # Best PPO config from the Bayesian search (trial 14).
        ppo_agent = PPO_agent(
            grid=args.grid,
            gamma=0.99,
            # gamma=0.999,
            gae_lambda=0.95,
            clip_epsilon=0.1881502205234746,
            policy_lr=0.00028701686449364406,
            value_lr=0.0010817190896430591,
            entropy_coef=0.03824745415070534,
            update_epochs=4,
            minibatch_size=128,
            rollout_steps=4096,
            hidden_sizes=(128, 128, 128),
            # hidden_sizes=(64,64),
            #reward_scale=1000.0,        # "high" reward goal magnitude
            max_grad_norm=5.0,
            activation="relu",
            seed=args.seed,
            device=args.device,
        )

        # Run PPO training with random starts (curriculum) -- same as how it was
        # tuned; evaluation is still from the fixed start_pos. Matches the DQN
        # training-start distribution for a fair comparison.
        # ppo_start_sampler = make_training_start_sampler(
        #     args.grid, start_pos, mode="random", seed=args.seed)
        ppo_full_train_agent, ppo_history, ppo_short_train_agent, ppo_mid_train_agent = train_ppo(
            agent=ppo_agent,
            env=env,
            max_steps_total=args.ppo_max_steps_total,
            short_train_steps_eval=args.ppo_short_train,
            mid_train_steps_eval=args.ppo_mid_train,
            max_steps_per_episode=args.ppo_max_steps_per_episode,
            start_pos=start_pos,
            seed=args.seed
        )
        
        training_convergence_plot(ppo_history, "PPO", args.results_dir, stamp, checking_window=50)

        ppo_train_successes = sum(1 for ep in ppo_history if ep["terminated"])
        ppo_metrics = {
            "train_total_successes": ppo_train_successes,
            "train_total_episodes": len(ppo_history),
            "train_success_rate": ppo_train_successes / len(ppo_history) if ppo_history else 0.0
        }

        # Run PPO evaluation and gather results
        ppo_full_eval = evaluate_ppo(
            agent=ppo_full_train_agent,
            grid=args.grid,
            max_steps_per_episode=args.ppo_eval_steps,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
            move_distance=args.move_distance,
            episodes=args.eval_episodes,
            reward_fn=EnvironmentContinuous._default_reward_function,
            optimal_steps=baseline_steps,
            agent_radius=args.agent_radius,
            turn_angle_deg=args.turn_angle_deg,
        )

        ppo_short_eval = evaluate_ppo(
            agent=ppo_short_train_agent,
            grid=args.grid,
            max_steps_per_episode=args.ppo_eval_steps,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
            move_distance=args.move_distance,
            episodes=args.eval_episodes,
            reward_fn=EnvironmentContinuous._default_reward_function,
            optimal_steps=baseline_steps,
            agent_radius=args.agent_radius,
            turn_angle_deg=args.turn_angle_deg,
        )

        ppo_mid_eval = evaluate_ppo(
            agent=ppo_mid_train_agent,
            grid=args.grid,
            max_steps_per_episode=args.ppo_eval_steps,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
            move_distance=args.move_distance,
            episodes=args.eval_episodes,
            reward_fn=EnvironmentContinuous._default_reward_function,
            optimal_steps=baseline_steps,
            agent_radius=args.agent_radius,
            turn_angle_deg=args.turn_angle_deg,
        )

        # Map PPO keys to standardized metrics keys
        ppo_eval_metrics = {
            "eval_total_reward": ppo_full_eval.get("total_reward", 0.0),
            "eval_steps": ppo_full_eval.get("avg_steps", 0.0),
            "eval_success_rate": ppo_full_eval.get("eval_success_rate", 0.0),
            "eval_spl": ppo_full_eval.get("SPL", 0.0),
            "eval_avg_collisions": ppo_full_eval.get("avg_failed_moves", 0.0),
            "short_train_eval_spl": ppo_short_eval.get("SPL", 0.0),
            "mid_train_eval_spl": ppo_mid_eval.get("SPL", 0.0),
        }

        # Redefine success as reaching within 20% of the optimal path length,
        # and report the rate of achieving the optimal path length.
        ppo_success_rate, ppo_optimal_rate = path_quality_rates(
            ppo_full_train_agent, args.grid, start_pos, args.eval_sigma, args.seed,
            args.eval_episodes, args.ppo_eval_steps)
        ppo_eval_metrics["eval_success_rate"] = ppo_success_rate
        ppo_eval_metrics["optimal_rate"] = ppo_optimal_rate

        # Collect data points
        ppo_steps = [args.ppo_short_train, args.ppo_mid_train, args.ppo_max_steps_total]
        ppo_spls  = [
            ppo_eval_metrics["short_train_eval_spl"],
            ppo_eval_metrics["mid_train_eval_spl"],
            ppo_eval_metrics["eval_spl"],
        ]


        # Sort by steps (x-axis) in case they aren't already ordered
        ppo_steps, ppo_spls = zip(*sorted(zip(ppo_steps, ppo_spls)))
        ppo_steps = [0] + list(ppo_steps)
        ppo_spls  = [0] + list(ppo_spls)

        # Calculate AUC using the trapezoidal rule
        ppo_auc = trapezoid(ppo_spls, ppo_steps)

        all_results["ppo"] = {**ppo_metrics, **ppo_eval_metrics, "agent": "PPO", "auc": ppo_auc}
        
        # save path image
        EnvironmentContinuous.evaluate_agent(
            grid_fp=args.grid,
            agent=ppo_agent,
            max_steps=args.ppo_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
            no_gui=args.no_gui
        )

        
    # Plot SPL vs #iterations to learn AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    if args.agents == "both":
        ax.plot(steps, spls, marker="o", linewidth=2, color="#1f77b4", label="DQN SPL")
        ax.fill_between(steps, spls, alpha=0.30, color="#1f77b4")  # 70% transparent = alpha 0.30
        ax.plot(ppo_steps, ppo_spls, marker="o", linewidth=2, color="#b04949", label="PPO SPL")
        ax.fill_between(ppo_steps, ppo_spls, alpha=0.30, color="#b04949")  # 70% transparent = alpha 0.30
        ax.set_title(f"SPL vs Training Iterations (DQN AUC = {dqn_auc:.1f}, PPO AUC = {ppo_auc:.1f})")
        ax.set_xlim(0, args.dqn_max_steps_total)
    if args.agents == "dqn":
        ax.plot(steps, spls, marker="o", linewidth=2, color="#1f77b4", label="DQN SPL")
        ax.fill_between(steps, spls, alpha=0.30, color="#1f77b4")  # 70% transparent = alpha 0.30
        ax.set_title(f"SPL vs Training Iterations (DQN AUC = {dqn_auc:.1f}")
        ax.set_xlim(0, args.dqn_max_steps_total)
    if args.agents == "ppo":
        ax.plot(ppo_steps, ppo_spls, marker="o", linewidth=2, color="#b04949", label="PPO SPL")
        ax.fill_between(ppo_steps, ppo_spls, alpha=0.30, color="#b04949")  # 70% transparent = alpha 0.30
        ax.set_title(f"SPL vs Training Iterations (PPO AUC = {ppo_auc:.1f})")
        ax.set_xlim(0, args.ppo_max_steps_total)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("SPL")
    ax.set_ylim(0, 1)
    
    
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
        

    # ── Save and Summary ─────────────────────────────────────────────────────
    if all_results:
        print_comparison(all_results)
        
        clean_results = json.loads(json.dumps(all_results, default=lambda o: str(o) if isinstance(o, Path) else o))
        combined_path = args.results_dir / f"combined_results_{stamp}.json"
        with open(combined_path, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"  Combined results saved to {combined_path}\n")
        # Save
        out_path = args.results_dir / f"{stamp}_auc_plot.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"AUC plot saved to {out_path.resolve()}")
        

if __name__ == "__main__":
    main()