"""Deep-pixel partial coverage and per-sample identity."""

import numpy as np
from numpy.typing import NDArray

from strata_datagen.exr import Deep
from strata_dataset import identity

VISIBLE = 0.02
OPAQUE = 0.98
LAYER_GAP_M = 0.05
FRONT_SLAB = 0.05


def partial_coverage(
    depth: NDArray[np.float32],
    alpha: NDArray[np.floating],
    counts: NDArray[np.int64],
    covered: NDArray[np.float64],
) -> NDArray[np.bool_]:
    size = counts.size
    pixel = np.repeat(np.arange(size), counts)
    starts = np.r_[np.int64(0), np.cumsum(counts)[:-1]]
    near = np.full(size, np.inf)
    populated = counts > 0
    near[populated] = np.minimum.reduceat(depth, starts[populated])
    front = depth <= near[pixel] * (1.0 + FRONT_SLAB) + LAYER_GAP_M
    with np.errstate(divide="ignore"):
        clear = np.bincount(pixel[front], weights=np.log1p(-np.minimum(alpha[front], 1.0)), minlength=size)
    front_cover = 1.0 - np.exp(clear)
    return (front_cover > VISIBLE) & (front_cover < OPAQUE) & (covered > VISIBLE)


IDENTITY_CHANNELS = ("instance_id", "instance_check", "instance_check2", "coverage")


def instance_ids(deep: Deep, order: NDArray[np.int64] | None = None) -> NDArray[np.int64]:
    """Per-sample instance ids; `identity.UNLABELLED` where the identity check fails."""
    return identity.decode(*(deep.channels[name] for name in IDENTITY_CHANNELS), order=order)
