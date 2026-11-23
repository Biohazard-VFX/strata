"""Lossless-over deep runs and sufficient statistics for the shared layer rule."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from strata_dataset import layers, roles, samples, segmented

MERGE_VERSION = "merge-1"
REBUILD_VERSION = "rebuild-1"

FIELDS = {
    "shape": "<i8",
    "offsets": "<i8",
    "z": "<f4",
    "zback": "<f4",
    "alpha": "<f8",
    "rgb": "<f8",
    "roles": "<f8",
    "ids": "<i8",
    "statistics": "<f8",
}
ROUNDOFF = 8 * np.finfo(np.float64).eps

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


def _arrays(path: Path) -> dict[str, NDArray[np.generic]]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(FIELDS):
            raise ValueError(f"merged arrays must be {sorted(FIELDS)}")
        return {name: data[name] for name in FIELDS}


def read(path: Path) -> Merged:
    """Read an owned merged frame without pickles; callers may concurrently read immutable files."""
    data = _arrays(path)
    shape = cast("tuple[int, int]", tuple(int(v) for v in data["shape"]))
    values = cast("RunArrays", tuple(data[name] for name in FIELDS if name != "shape"))
    return Merged(shape, *values)


def load_checked(path: Path, shape: tuple[int, int]) -> tuple[dict[str, NDArray[np.generic]], list[str]]:
    """Owned arrays and format violations, read once; immutable inputs and thread-safe calls."""
    try:
        arrays = _arrays(path)
    except (OSError, ValueError, KeyError, BadZipFile, EOFError, TypeError) as error:
        return {}, [f"merged {path.parent.name}: unreadable: {error}"]
    found = [
        f"{name}: expected {dtype} C-order"
        for name, dtype in FIELDS.items()
        if arrays[name].dtype.str != dtype or not arrays[name].flags.c_contiguous
    ]
    if arrays["shape"].shape != (2,) or arrays["shape"].tolist() != list(shape):
        found.append("shape differs from the camera frame")
    if not found:
        found += samples.sample_errors(arrays, shape[0] * shape[1], roundoff=float(ROUNDOFF))
    count = arrays["z"].size
    if arrays["zback"].shape != (count,) or arrays["statistics"].shape != (count, 4):
        found.append("zback or statistics shape differs from the sample count")
    if not found:
        back = cast("NDArray[np.float32]", arrays["zback"])
        z = cast("NDArray[np.float32]", arrays["z"])
        statistics = cast("NDArray[np.float64]", arrays["statistics"])
        if not np.isfinite(back).all() or np.any(back < z):
            found.append("zback must be finite and at least z")
        overlap = back[:-1] > z[1:]
        boundaries = arrays["offsets"].astype(np.int64)[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < count)]
        overlap[boundaries - 1] = False
        if np.any(overlap):
            found.append("merged intervals overlap")
        if not np.isfinite(statistics).all() or np.any(statistics < 0) or np.any(statistics[:, 0] > 1):
            found.append("statistics must be finite and nonnegative, with transmittance at most one")
    return arrays, [f"merged {path.parent.name}: {error}" for error in found]


def errors(path: Path, shape: tuple[int, int]) -> list[str]:
    """Read and validate immutable merged deep; return violations without mutating borrowed files."""
    return load_checked(path, shape)[1]
