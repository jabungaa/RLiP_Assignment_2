"""Visualize a grid config file.

Usage:
    python visualize_grid.py
    python visualize_grid.py --grid grid_configs/A1_grid.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Cell value → display colour and label
CELL_STYLE = {
    0: ("#FFFFFF", "empty"),
    1: ("#222222", "boundary"),
    2: ("#666666", "obstacle"),
    3: ("#4CAF50", "target"),
    4: ("#E1602F", "start"),
}


def visualize(grid_fp: Path, save: bool = True):
    grid = np.load(grid_fp)          # shape (n_cols, n_rows), grid[col, row]
    n_cols, n_rows = grid.shape

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.6), max(4, n_rows * 0.6)))

    for col in range(n_cols):
        for row in range(n_rows):
            val = int(grid[col, row])
            color, _ = CELL_STYLE.get(val, ("#FF00FF", "unknown"))
            rect = mpatches.Rectangle(
                (col, row), 1, 1,
                facecolor=color, edgecolor="#AAAAAA", linewidth=0.5,
            )
            ax.add_patch(rect)

            # Label start and target cells
            if val == 4:
                ax.text(col + 0.5, row + 0.5, "S", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            elif val == 3:
                ax.text(col + 0.5, row + 0.5, "T", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(n_cols + 1))
    ax.set_yticks(range(n_rows + 1))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("col (x)")
    ax.set_ylabel("row (y)")
    ax.set_title(f"{grid_fp.name}  ({n_cols}×{n_rows})")

    legend = [
        mpatches.Patch(color=color, label=label)
        for val, (color, label) in CELL_STYLE.items()
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()

    if save:
        out = Path(__file__).parent / "results" / (grid_fp.stem + "_grid.png")
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a grid config.")
    parser.add_argument("--grid", type=Path,
                        default=Path("grid_configs/restaurant_small.npy"))
    parser.add_argument("--no_save", action="store_true",
                        help="Skip saving the image to results/.")
    args = parser.parse_args()
    visualize(args.grid, save=not args.no_save)


if __name__ == "__main__":
    main()
