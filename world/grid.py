"""Grid.

Credit to Tom v. Meer for writing this.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


class Grid:

    CELL_TYPES = {
            "empty": 0,
            "boundary": 1,
            "obstacle": 2,
            "target": 3,
            "start": 4
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
    def load_grid(grid_file_path: Path) -> Grid:
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