"""
Training and evaluation pipeline by importing existing functionalities from PPO and DQN
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Pipeline imports
from train_dqn import train_DQN, evaluate_DQN
from test_ppo import train_ppo, evaluate_ppo

# PPO-specific imports needed for instantiation
from agents.PPO import PPO_agent
from world.environment_continuous import EnvironmentContinuous
from world.path_visualizer import visualize_path, save_path_image

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
    parser.add_argument("--no_gui", default=True, help="Disable GUI during training.")
    parser.add_argument("--eval_gui", action="store_true", default=True, help="Enable GUI during evaluation.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes for evaluation.")

    # DQN Hyperparameters
    dqn = parser.add_argument_group("DQN")
    dqn.add_argument("--dqn_episodes", type=int, default=5)
    dqn.add_argument("--dqn_max_steps", type=int, default=50)
    dqn.add_argument("--dqn_lr", type=float, default=0.001)
    dqn.add_argument("--dqn_gamma", type=float, default=0.99)

    # PPO Hyperparameters
    ppo = parser.add_argument_group("PPO")
    ppo.add_argument("--ppo_episodes", type=int, default=5)
    ppo.add_argument("--ppo_iters", type=int, default=200)
    ppo.add_argument("--ppo_eval_steps", type=int, default=50)
    ppo.add_argument("--ppo_policy_lr", type=float, default=3e-4)
    ppo.add_argument("--ppo_value_lr", type=float, default=1e-3)
    
    parser.add_argument("--device", type=str, default="cuda")
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
        dqn_agent, dqn_history = train_DQN(
            grid=args.grid,
            n_episodes=args.dqn_episodes,
            max_steps_per_episode=args.dqn_max_steps,
            sigma=args.sigma,
            learning_rate=args.dqn_lr,
            gamma=args.dqn_gamma,
            random_seed=args.seed,
            agent_start_pos=start_pos,
            no_gui=True,
            device=args.device
        )
        
        dqn_train_successes = sum(1 for ep in dqn_history if ep["terminated"])
        dqn_metrics = {
            "train_total_successes": dqn_train_successes,
            "train_total_episodes": len(dqn_history),
            "train_success_rate": dqn_train_successes / len(dqn_history) if dqn_history else 0.0
        }
        
        # Evaluate DQN and gather results
        dqn_eval = evaluate_DQN(
            agent=dqn_agent,
            grid=args.grid,
            max_steps_per_episode=args.dqn_max_steps,
            sigma=args.eval_sigma,
            agent_start_pos=start_pos,
            no_gui=True, 
            random_seed=args.seed
        )
        
        dqn_eval_metrics = {
            "eval_total_reward": dqn_eval["total_reward"],
            "eval_steps": dqn_eval["steps"],
            "eval_success_rate": 1.0 if dqn_eval["terminated"] else 0.0
        }
        
        all_results["dqn"] = {**dqn_metrics, **dqn_eval_metrics, "agent": "DQN"}

        # save path image
        EnvironmentContinuous.evaluate_agent(
            grid_fp=args.grid,
            agent=dqn_agent,
            max_steps=args.dqn_max_steps,
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
            network_mode="separate",
            reward_scale=100.0,
            activation="tanh",
            seed=args.seed,
            device=args.device,
        )
        
        # Run PPO training
        ppo_metrics = train_ppo(
            agent=ppo_agent,
            env=env,
            episodes=args.ppo_episodes,
            iters=args.ppo_iters,
            start_sampler=lambda: start_pos,
            repeat_visit_penalty=0.0
        )
        
        # Run PPO evaluation and gather results
        ppo_eval = evaluate_ppo(
            agent=ppo_agent,
            Environment=EnvironmentContinuous, 
            grid_fp=args.grid,
            reward_fn=EnvironmentContinuous._high_reward_function,
            start_pos=start_pos,
            sigma=args.eval_sigma,
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.ppo_eval_steps,
            gamma=0.999
        )
        
        # Map PPO keys to standardized metrics keys
        ppo_eval_metrics = {
            "eval_total_reward": ppo_eval.get("eval_avg_reward", 0.0),
            "eval_steps": ppo_eval.get("eval_avg_steps", 0.0),
            "eval_success_rate": ppo_eval.get("eval_success_rate", 0.0)
        }
        
        all_results["ppo"] = {**ppo_metrics, **ppo_eval_metrics, "agent": "PPO"}

    # ── Save and Summary ─────────────────────────────────────────────────────
    if all_results:
        print_comparison(all_results)
        
        clean_results = json.loads(json.dumps(all_results, default=lambda o: str(o) if isinstance(o, Path) else o))
        combined_path = args.results_dir / f"combined_results_{stamp}.json"
        with open(combined_path, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"  Combined results saved to {combined_path}\n")

if __name__ == "__main__":
    main()