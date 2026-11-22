"""K-layer target rule shared by data generation and training.

Split at the largest K-1 log-depth gaps above log(gap_ratio), first tie first.
Store conditional premultiplied color, alpha and visible-weighted inverse depth.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from strata_dataset import segmented

Array = NDArray[np.float64]
LAYERS = 4
GAP_RATIO = 1.15


@dataclass(frozen=True)
class Layers:
    """inverse_depth and alpha are (K, H, W); rgb is (K, 3, H, W)."""

    inverse_depth: NDArray[np.float32]
    alpha: NDArray[np.float32]
    rgb: NDArray[np.float32]


def reduce_samples(
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: NDArray[np.int64],
    shape: tuple[int, int],
    *,
    layers: int = LAYERS,
    gap_ratio: float = GAP_RATIO,
    sky: NDArray[np.bool_] | None = None,
) -> Layers:
    """Own K-layer arrays; borrowed inputs are immutable and calls are thread-safe.

    Samples are pixel-major, in any depth order. RGB is premultiplied.
    Optional sky membership masks geometry of any cluster with visible sky.
    """
    pixels = shape[0] * shape[1]
    if layers < 1 or gap_ratio <= 1:
        raise ValueError("layers must be positive and gap_ratio must exceed one")
    if counts.size != pixels or int(counts.sum()) != depth.size or rgb.shape != (depth.size, 3):
        raise ValueError("deep channel sizes must agree with sample counts and image shape")
    if np.any(counts < 0) or not (np.isfinite(depth).all() and np.all(depth > 0)):
        raise ValueError("sample counts must be nonnegative and depths finite and positive")
    if alpha.shape != depth.shape or (sky is not None and sky.shape != depth.shape):
        raise ValueError("sample alpha and sky shapes must match depth")
    output = np.zeros((pixels * layers, 5), dtype=np.float64)
    if depth.size:
        segmented.fill(
            output,
            depth,
            alpha,
            rgb,
            counts,
            np.zeros(depth.size, dtype=np.bool_) if sky is None else sky,
            layers,
            float(np.log(gap_ratio)),
        )
    cube = output.reshape(*shape, layers, 5).transpose(2, 3, 0, 1).astype(np.float32)
    return Layers(cube[:, 0], np.clip(cube[:, 1], 0, 1), cube[:, 2:])
