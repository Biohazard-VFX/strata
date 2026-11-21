"""Ground data shared between generators: a regular heightfield plus named [0, 1] masks."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Grid = NDArray[np.float64]


@dataclass(frozen=True)
class Ground:
    """Heights and masks on an (n, n) grid; row index runs along +z, column index along +x."""

    heights: Grid
    cell_m: float
    masks: Mapping[str, Grid]

    @property
    def half_size_m(self) -> float:
        return 0.5 * self.cell_m * (self.heights.shape[0] - 1)

    def _bilinear(self, grid: Grid, x: Grid, z: Grid) -> Grid:
        last = grid.shape[0] - 1
        u = np.clip((x + self.half_size_m) / self.cell_m, 0.0, last)
        v = np.clip((z + self.half_size_m) / self.cell_m, 0.0, last)
        i0 = np.minimum(np.floor(u).astype(np.int64), last - 1)
        j0 = np.minimum(np.floor(v).astype(np.int64), last - 1)
        fu, fv = u - i0, v - j0
        top = grid[j0, i0] * (1 - fu) + grid[j0, i0 + 1] * fu
        bottom = grid[j0 + 1, i0] * (1 - fu) + grid[j0 + 1, i0 + 1] * fu
        return top * (1 - fv) + bottom * fv

    def height(self, x: Grid, z: Grid) -> Grid:
        return self._bilinear(self.heights, x, z)

    def mask(self, name: str, x: Grid, z: Grid) -> Grid:
        return self._bilinear(self.masks[name], x, z)

    def normal(self, x: Grid, z: Grid) -> Grid:
        """Unit normals, shape (..., 3), from central differences of the interpolated surface."""
        step = self.cell_m
        dx = (self.height(x + step, z) - self.height(x - step, z)) / (2 * step)
        dz = (self.height(x, z + step) - self.height(x, z - step)) / (2 * step)
        n = np.stack([-dx, np.ones_like(dx), -dz], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)
