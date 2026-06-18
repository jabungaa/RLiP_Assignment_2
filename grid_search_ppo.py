"""Grid search over PPO hyperparameters.

Runs test_ppo.py for every combination in PARAM_GRID (or a random subset),
collects the JSON metrics each run writes to results/, and prints a ranked
summary at the end.

Usage:
    python grid_search_ppo.py                        # full grid
    python grid_search_ppo.py --max_runs 20          # random sample of 20
    python grid_search_ppo.py --max_runs 20 --seed 7 # reproducible sample
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Hyperparameter grid — edit these lists to define the search space
# ---------------------------------------------------------------------------
PARAM_GRID: dict[str, list] = {
    "policy_lr":     [1e-4, 3e-4, 1e-3],
    "value_lr":      [1e-4, 3e-4, 1e-3],
    "entropy_coef":  [0.001, 0.01, 0.05],
    "rollout_steps": [1024, 4096],
    "minibatch_size":[64, 256],
    "hidden_sizes":  ["64,64", "128,128"],
}

# Fixed settings shared by every run — override as needed
FIXED_ARGS: dict[str, str] = {
    "grid":             "grid_configs/A1_grid.npy",
    "reward":           "default",
    "episodes":         "500",
    "iters":            "1000",
    "train_start_mode": "random",
    "eval_episodes":    "3",
    "eval_max_steps":   "1000",
    "gamma":            "0.999",
    "seed":             "0",
}

# Metric used to rank runs (must be a key in the saved JSON)
RANK_BY = "eval_success_rate"
# ---------------------------------------------------------------------------


def all_combinations(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def build_cmd(combo: dict, fixed: dict) -> list[str]:
    cmd = [sys.executable, "test_ppo.py"]
    for k, v in {**fixed, **combo}.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def latest_json(results_dir: Path, after: float) -> dict | None:
    """Return the most recently written *_metrics.json created after `after`."""
    candidates = sorted(
        [p for p in results_dir.glob("*_metrics.json") if p.stat().st_mtime > after],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def print_summary(results: list[dict], rank_by: str):
    ranked = sorted(results, key=lambda r: r.get(rank_by, -1), reverse=True)
    print(f"\n{'='*100}")
    print(f"GRID SEARCH SUMMARY  —  ranked by {rank_by}  ({len(ranked)} runs)")
    print(f"{'='*100}")
    header_keys = list(PARAM_GRID.keys())
    col_w = 10
    header = (
        f"{'rank':>4}  {'eval_succ':>9}  {'train_succ':>10}  {'train_l100':>10}  "
        + "  ".join(f"{k[:col_w]:>{col_w}}" for k in header_keys)
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked, 1):
        eval_score  = r.get(rank_by, float("nan"))
        train_score = r.get("train_success_rate", float("nan"))
        train_l100  = r.get("train_success_rate_last_100", float("nan"))
        combo_str = "  ".join(f"{str(r['combo'].get(k,''))[:col_w]:>{col_w}}" for k in header_keys)
        print(f"{i:>4}  {eval_score:>9.4f}  {train_score:>10.4f}  {train_l100:>10.4f}  {combo_str}")
    print()
    if ranked:
        best = ranked[0]
        print("Best combination:")
        for k, v in best["combo"].items():
            print(f"  --{k} {v}")
        print(f"  {rank_by}: {best.get(rank_by, 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="PPO hyperparameter grid search.")
    parser.add_argument("--max_runs", type=int, default=None,
                        help="Randomly sample this many combinations. Default: full grid.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random sampling. Default: 0.")
    args = parser.parse_args()

    combos = all_combinations(PARAM_GRID)
    total = len(combos)

    if args.max_runs is not None and args.max_runs < total:
        random.seed(args.seed)
        combos = random.sample(combos, args.max_runs)
        print(f"Random search: {len(combos)}/{total} combinations (seed={args.seed})")
    else:
        print(f"Full grid search: {total} combinations")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results: list[dict] = []

    for run_idx, combo in enumerate(combos, 1):
        print(f"\n[{run_idx}/{len(combos)}] {combo}")
        cmd = build_cmd(combo, FIXED_ARGS)
        t0 = time.time()
        before = time.time()

        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
        )

        elapsed = time.time() - t0
        metrics = latest_json(results_dir, after=before)

        if proc.returncode != 0 or metrics is None:
            print(f"  FAILED (exit {proc.returncode}, elapsed {elapsed:.0f}s)")
            all_results.append({"combo": combo, RANK_BY: -1.0})
            continue

        score = metrics.get(RANK_BY, float("nan"))
        print(f"  {RANK_BY}={score:.4f}  elapsed={elapsed:.0f}s")
        all_results.append({"combo": combo, **metrics})

    summary_path = results_dir / f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}_grid_search.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")

    print_summary(all_results, RANK_BY)


if __name__ == "__main__":
    main()
