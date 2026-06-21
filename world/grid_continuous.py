"""Continuous grid.

An extension of :class:`world.grid.Grid` that converts the discrete grid into a
continuous 2D world described by a set of wall line segments tracing the
*outlines* of the walls and obstacles.
Run this file to test the geometry extraction and visualization.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Grid:
    """Discrete grid representation of the world as a 2D integer array.

    Inlined here (rather than imported) so this module is a self-contained
    drop-in replacement for the original ``grid.py``. Credit to Tom v. Meer for
    the original.
    """

    CELL_TYPES = {
        "empty": 0,
        "boundary": 1,
        "obstacle": 2,
        "target": 3,
        "start": 4,
    }

    def __init__(self, n_cols: int, n_rows: int):
        """Grid representation of the world 
           as a 2D numpy integer array.

        Possible grid values are:
        - Empty: 0,
        - Boundary: 1,
        - Obstacle: 2,
        - Dirt: 3
        - Start: 4

        Args:
            n_cols: Number of grid columns.
            n_rows: Number of grid rows.
        """

        # Building the boundary of the grid:
        self.cells = np.zeros((n_cols, n_rows), dtype=np.int8)
        self.cells[0, :] = self.cells[-1, :] = 1
        self.cells[:, 0] = self.cells[:, -1] = 1

        self.n_rows = self.cells.shape[1]
        self.n_cols = self.cells.shape[0]

    def place_object(self, x, y, object_type):
        """Places an object on the grid.

        Args:
            x: x-coordinate of the object.
            y: y-coordinate of the object.
            object_type: Type of the object.
        """
        self.cells[x][y] = self.CELL_TYPES[object_type]

    @staticmethod
    def load_grid(grid_file_path: Path) -> "Grid":
        """Loads a numpy array from file path.

        Returns:
            A Grid object from the file.
        """
        grid_array = np.load(grid_file_path)
        grid = Grid(grid_array.shape[0], grid_array.shape[1])
        grid.cells = grid_array
        return grid

    def save_grid_file(self, grid_file_path: Path):
        """Saves the numpy array representation of 
        the grid to file path.

        Args:
            grid_file_path: File path where the grid file is to be saved.
        """
        np.save(grid_file_path.with_suffix(".npy"), self.cells)


SOLID_VALUES = (1, 2)  # 1 = boundary, 2 = obstacle

Point = tuple[int, int]
Segment = tuple[tuple[float, float], tuple[float, float]]


class GridContinuous(Grid):
    """Grid that exposes a continuous outline (boundary-segment) view."""

    @classmethod
    def from_cells(cls, cells: np.ndarray) -> "GridContinuous":
        """Builds an instance directly from a cell array."""
        obj = cls(cells.shape[0], cells.shape[1])
        obj.cells = cells
        return obj

    @classmethod
    def from_file(cls, grid_fp) -> "GridContinuous":
        """Loads a grid array from disk and wraps it."""
        return cls.from_cells(np.load(grid_fp))

    # ------------------------------------------------------------------ #
    # Geometry extraction
    # ------------------------------------------------------------------ #
    def solid_mask(self) -> np.ndarray:
        """Boolean mask, True where a cell is a wall or obstacle."""
        mask = np.zeros(self.cells.shape, dtype=bool)
        for v in SOLID_VALUES:
            mask |= self.cells == v
        return mask

    def _boundary_edges(self, mask: np.ndarray):
        """Collects unit edges that separate a solid cell from free space.

        Returns:
            (vertical_edges, horizontal_edges) as sets of (p1, p2) integer
            corner pairs, with p1 < p2 so duplicates collapse.
        """
        n_cols, n_rows = mask.shape
        vertical: set[tuple[Point, Point]] = set()    # constant x
        horizontal: set[tuple[Point, Point]] = set()  # constant y

        def solid(c: int, r: int) -> bool:
            if c < 0 or c >= n_cols or r < 0 or r >= n_rows:
                return False
            return bool(mask[c, r])

        for c, r in np.argwhere(mask):
            c, r = int(c), int(r)
            if not solid(c - 1, r):  # left face
                vertical.add(((c, r), (c, r + 1)))
            if not solid(c + 1, r):  # right face
                vertical.add(((c + 1, r), (c + 1, r + 1)))
            if not solid(c, r - 1):  # top face
                horizontal.add(((c, r), (c + 1, r)))
            if not solid(c, r + 1):  # bottom face
                horizontal.add(((c, r + 1), (c + 1, r + 1)))

        return vertical, horizontal

    @staticmethod
    def _merge_collinear(edges, axis: int) -> list[Segment]:
        """Merges contiguous collinear unit edges into long segments.

        Args:
            edges: Unit edges, all parallel to ``axis``.
            axis: 0 if edges vary along x (horizontal), 1 if along y (vertical).
        """
        const_axis = 1 - axis
        lines: dict[int, list[tuple[int, int]]] = {}
        for p1, p2 in edges:
            line = p1[const_axis]
            a, b = sorted((p1[axis], p2[axis]))
            lines.setdefault(line, []).append((a, b))

        merged: list[Segment] = []
        for line, intervals in lines.items():
            intervals.sort()
            cur_start, cur_end = intervals[0]
            for a, b in intervals[1:]:
                if a <= cur_end:
                    cur_end = max(cur_end, b)
                else:
                    merged.append(GridContinuous._make_segment(
                        line, cur_start, cur_end, axis))
                    cur_start, cur_end = a, b
            merged.append(GridContinuous._make_segment(
                line, cur_start, cur_end, axis))
        return merged

    @staticmethod
    def _make_segment(line: int, start: int, end: int, axis: int) -> Segment:
        if axis == 0:  # horizontal: x varies, y = line
            return ((float(start), float(line)), (float(end), float(line)))
        return ((float(line), float(start)), (float(line), float(end)))

    def to_segments(self) -> list[Segment]:
        """Converts the grid into continuous-space wall outline segments."""
        mask = self.solid_mask()
        vertical, horizontal = self._boundary_edges(mask)
        segments = self._merge_collinear(horizontal, axis=0)
        segments += self._merge_collinear(vertical, axis=1)
        return segments

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def visualize(self, save_path: str | None = None, show: bool = False):
        """Draws the solid cells and the extracted continuous wall outline."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        segments = self.to_segments()
        n_cols, n_rows = self.cells.shape
        fig, ax = plt.subplots(figsize=(7, 7))

        mask = self.solid_mask()
        for c, r in np.argwhere(mask):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="0.85",
                                   edgecolor="none"))
        for (x1, y1), (x2, y2) in segments:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2,
                    solid_capstyle="round")
        for (x1, y1), (x2, y2) in segments:
            ax.plot([x1, x2], [y1, y2], "o", color="crimson", markersize=4)

        # Start (cell value 4) and end/target (cell value 3) positions, drawn
        # at the centre of their cells.
        for c, r in np.argwhere(self.cells == 4):
            ax.plot(c + 0.5, r + 0.5, marker="o", markersize=12,
                    markerfacecolor="#2ca02c", markeredgecolor="black",
                    linestyle="none", label="start", zorder=5)
        for c, r in np.argwhere(self.cells == 3):
            ax.plot(c + 0.5, r + 0.5, marker="*", markersize=18,
                    markerfacecolor="#1f77b4", markeredgecolor="black",
                    linestyle="none", label="end", zorder=5)

        ax.set_xlim(-0.5, n_cols + 0.5)
        ax.set_ylim(-0.5, n_rows + 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"v2 outline: {len(segments)} wall segments")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_xticks(range(n_cols + 1))
        ax.set_yticks(range(n_rows + 1))

        # De-duplicate legend labels (one entry per type even with many cells).
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper right")

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    grid_fp = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else repo_root / "grid_configs" / "restaurant_test.npy"
    grid = GridContinuous.from_file(grid_fp)
    segs = grid.to_segments()
    print(f"[v2] {grid_fp.name} {grid.cells.shape}: {len(segs)} wall segments")
    out = grid_fp.with_suffix(".v2.png")
    grid.visualize(save_path=str(out))
    print(f"Saved visualization to {out}")
