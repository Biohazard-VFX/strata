"""Lossless-over deep runs and sufficient statistics for the shared layer rule."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from strata_dataset import layers, roles, segmented

RunArrays = tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]


@dataclass(frozen=True)
class Merged:
    """Owned, sorted deep runs; immutable borrowed inputs and thread-safe operations."""

    shape: tuple[int, int]
    offsets: NDArray[np.int64]
    z: NDArray[np.float32]
    zback: NDArray[np.float32]
    alpha: NDArray[np.float64]
    rgb: NDArray[np.float64]
    roles: NDArray[np.float64]
    ids: NDArray[np.int64]
    statistics: NDArray[np.float64]

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "shape": np.array(self.shape, np.int64),
            "offsets": self.offsets,
            "z": self.z,
            "zback": self.zback,
            "alpha": self.alpha,
            "rgb": self.rgb,
            "roles": self.roles,
            "ids": self.ids,
            "statistics": self.statistics,
        }

    def targets(self, count: int = layers.LAYERS) -> layers.Layers:
        return layers.reduce_samples(
            self.z,
            self.alpha,
            self.rgb,
            np.diff(self.offsets),
            self.shape,
            layers=count,
            zback=self.zback,
            statistics=self.statistics,
        )


def merge(
    shape: tuple[int, int],
    offsets: NDArray[np.int64],
    z: NDArray[np.float32],
    alpha: NDArray[np.float32],
    rgb: NDArray[np.float32],
    *,
    semantic: NDArray[np.float32],
    ids: NDArray[np.int64],
) -> Merged:
    """Composite adjacent same-id, sub-threshold runs; visibility-discarded samples stay singleton."""
    values = cast(
        "RunArrays",
        segmented.merge_runs(
            z,
            alpha,
            rgb,
            semantic,
            ids,
            np.diff(offsets),
            roles.sky(alpha, semantic[:, 0]),
            float(np.log(layers.GAP_RATIO)),
        ),
    )
    return Merged(shape, *values)


def read(path: Path) -> Merged:
    """Read an owned merged frame without pickles; callers may concurrently read immutable files."""
    with np.load(path, allow_pickle=False) as data:
        shape = cast("tuple[int, int]", tuple(int(v) for v in data["shape"]))
        return Merged(
            shape,
            data["offsets"],
            data["z"],
            data["zback"],
            data["alpha"],
            data["rgb"],
            data["roles"],
            data["ids"],
            data["statistics"],
        )
