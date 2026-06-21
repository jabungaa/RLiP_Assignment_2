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

# Pipeline imports
from train_dqn import train_DQN, evaluate_DQN
# from test_ppo import train_ppo, evaluate_ppo
from train_ppo_functions import train_ppo, evaluate_ppo

# PPO-specific imports needed for instantiation
from agents.PPO import PPO_agent
from world.environment_continuous import EnvironmentContinuous
from world.path_visualizer import visualize_path, save_path_image
import torch

print("FILE LOADED")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DQN and/or PPO agents.")

    # Shared configurations
    parser.add_argument("--grid", type=Path, default=Path("grid_configs/A1_grid.npy"), help="Path to the .npy grid file.")
    parser.add_argument("--agents", choices=("dqn", "ppo", "both"), default="both", help="Which agent(s) to run.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Environment stochasticity (training).")
    parser.add_argument("--eval_sigma", type=float, default=0.0, help="Environment stochasticity (evaluation).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_pos", type=str, default=None, help="Agent start position as row,col.")
    parser.add_argument("--no_gui", action="store_true", default=False,
                    help="Disable GUI during training.")
    parser.add_argument("--eval_gui", action="store_true", default=True, help="Enable GUI during evaluation.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes for evaluation.")

    # DQN Hyperparameters
    dqn = parser.add_argument_group("DQN")
    dqn.add_argument("--dqn_episodes", type=int, default=5)
    dqn.add_argument("--dqn_max_steps_total", type=int, default=200000)
    dqn.add_argument("--dqn_short_train", type=int, default=50000)
    dqn.add_argument("--dqn_mid_train", type=int, default=100000)
    dqn.add_argument("--dqn_max_steps_per_episode", type=int, default=50)
    dqn.add_argument("--dqn_lr", type=float, default=0.001)
    dqn.add_argument("--dqn_gamma", type=float, default=0.99)

    # PPO Hyperparameters
    ppo = parser.add_argument_group("PPO")
    # ppo.add_argument("--ppo_episodes", type=int, default=5)
    ppo.add_argument("--ppo_max_steps_total", type=int, default=200000)
    ppo.add_argument("--ppo_short_train", type=int, default=50000)
    ppo.add_argument("--ppo_mid_train", type=int, default=100000)
    ppo.add_argument("--ppo_max_steps_per_episode", type=int, default=1000)
    ppo.add_argument("--ppo_eval_steps", type=int, default=50)
    ppo.add_argument("--ppo_policy_lr", type=float, default=3e-4)
    ppo.add_argument("--ppo_value_lr", type=float, default=1e-3)
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"\n{'─'*60}\n  Summary — Training & Evaluation\n{'─'*60}")
    agents = list(results.keys())
    print(f"  {'Metric':<25}" + "".join(f"{a.upper():>15}" for a in agents))
    print("  " + "─" * (25 + 15 * len(agents)))
    
    metrics = [
        ("train_success_rate", "Train Success Rate"),
        ("eval_success_rate", "Eval Success Rate"),
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

def main():
    print("ENTERED MAIN")
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    start_pos = parse_start_pos(args.start_pos, args.grid)

    all_results = {}
    print("agents =", args.agents)

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
            sigma=args.sigma,
            learning_rate=args.dqn_lr,
            gamma=args.dqn_gamma,
            random_seed=args.seed,
            agent_start_pos=start_pos,
            no_gui=args.no_gui,
            device=args.device
        )
        
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
            episodes=args.eval_episodes
        )

        mid_train_dqn_eval = evaluate_DQN(
            agent=mid_train_agent,
            grid=args.grid,
            max_steps_per_episode=args.dqn_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            no_gui=True,
            episodes=args.eval_episodes
        )

        dqn_eval = evaluate_DQN(
            agent=dqn_agent,
            grid=args.grid,
            max_steps_per_episode=args.dqn_max_steps_per_episode,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            no_gui=True,
            episodes=args.eval_episodes
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

        env = EnvironmentContinuous(
            grid_fp=args.grid,
            no_gui=args.no_gui,
            sigma=args.sigma,
            agent_start_pos=start_pos,
            random_seed=args.seed,
        )
        
        ppo_agent = PPO_agent(
            grid=args.grid,
            gamma=0.999,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            policy_lr=args.ppo_policy_lr,
            value_lr=args.ppo_value_lr,
            entropy_coef=0.01,
            update_epochs=4,
            rollout_steps=128,
            hidden_sizes=(64, 128),
            reward_scale=100.0,
            activation="tanh",
            seed=args.seed,
            device=args.device,
        )

        # Run PPO training
        ppo_full_train_agent, ppo_history, ppo_short_train_agent, ppo_mid_train_agent = train_ppo(
            agent=ppo_agent,
            env=env,
            max_steps_total=args.ppo_max_steps_total,
            short_train_steps_eval=args.ppo_short_train,
            mid_train_steps_eval=args.ppo_mid_train,
            max_steps_per_episode=args.ppo_max_steps_per_episode,
            start_pos=start_pos
            # repeat_visit_penalty=0.0
        )
        
        ppo_train_successes = sum(1 for ep in ppo_history if ep["terminated"])
        ppo_metrics = {
            "train_total_successes": ppo_train_successes,
            "train_total_episodes": len(ppo_history),
            "train_success_rate": ppo_train_successes / len(ppo_history) if ppo_history else 0.0
        }

        # Run PPO evaluation and gather results
        ppo_full_eval = evaluate_ppo(
            agent=ppo_full_train_agent,
            Environment=EnvironmentContinuous, 
            grid_fp=args.grid,
            reward_fn=EnvironmentContinuous._default_reward_function,
            start_pos=start_pos,
            sigma=args.eval_sigma,
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.ppo_eval_steps,
            # gamma=0.999
        )

        ppo_short_eval = evaluate_ppo(
            agent=ppo_short_train_agent,
            Environment=EnvironmentContinuous, 
            grid_fp=args.grid,
            reward_fn=EnvironmentContinuous._default_reward_function,
            start_pos=start_pos,
            sigma=args.eval_sigma,
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.ppo_eval_steps,
            # gamma=0.999
        )

        ppo_mid_eval = evaluate_ppo(
            agent=ppo_mid_train_agent,
            Environment=EnvironmentContinuous, 
            grid_fp=args.grid,
            reward_fn=EnvironmentContinuous._default_reward_function,
            start_pos=start_pos,
            sigma=args.eval_sigma,
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.ppo_eval_steps,
            # gamma=0.999
        )

        # Map PPO keys to standardized metrics keys
        ppo_eval_metrics = {
            "eval_total_reward": ppo_full_eval.get("total_reward", 0.0),
            "eval_steps": ppo_full_eval.get("eval_avg_steps", 0.0),
            "eval_success_rate": ppo_full_eval.get("eval_success_rate", 0.0),
            "eval_spl": ppo_full_eval.get("eval_spl", 0.0),
            "eval_avg_collisions":ppo_full_eval.get("eval_average_failed_moves", 0.0), 
            "short_train_eval_spl": ppo_short_eval.get("eval_spl", 0.0),
            "mid_train_eval_spl": ppo_mid_eval.get("eval_spl", 0.0),
        }
        
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