"""Deep-pixel measurements: front-to-back flattening, partial coverage and per-sample identity."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from strata_datagen.exr import Deep
from strata_dataset import identity, segmented

VISIBLE = 0.02
OPAQUE = 0.98
LAYER_GAP_M = 0.05
FRONT_SLAB = 0.05


@dataclass(frozen=True)
class Flattened:
    rgba: NDArray[np.float64]
    alpha: NDArray[np.float64]
    partial: NDArray[np.bool_]


def _segment_cumsum(
    values: NDArray[np.float64],
    starts: NDArray[np.int64],
    counts: NDArray[np.int64],
    cumulative: float,
) -> tuple[NDArray[np.float64], float]:
    """Exclusive sums with the whole-frame accumulation order carried across rows."""
    total = values.copy()
    if values.size:
        total[0] += cumulative
        np.cumsum(total, out=total)
        cumulative = float(total[-1])
    total -= values
    total -= np.repeat(total[starts[counts > 0]], counts[counts > 0])
    return total, cumulative


def _flatten_row(deep: Deep, cumulative: float) -> tuple[Flattened, float]:
    order = segmented.order(deep.channels["Z"], deep.counts)
    pixel = deep.pixel_index()
    alpha = deep.channels["A"][order].astype(np.float64)
    np.clip(alpha, 0.0, 0.999999, out=alpha)
    starts = deep.offsets[:-1]
    transmittance, cumulative = _segment_cumsum(np.log1p(-alpha), starts, deep.counts, cumulative)
    np.exp(transmittance, out=transmittance)
    size = deep.counts.size
    rgba = np.zeros((size, 4))
    for i, name in enumerate("RGBA"):
        rgba[:, i] = np.bincount(pixel, weights=deep.channels[name][order] * transmittance, minlength=size)
    depth = deep.channels["Z"][order].astype(np.float64)
    near = np.full(size, np.inf)
    populated = deep.counts > 0
    near[populated] = np.minimum.reduceat(depth, starts[populated])
    front = depth <= near[pixel] * (1.0 + FRONT_SLAB) + LAYER_GAP_M
    clear = np.bincount(pixel[front], weights=np.log1p(-alpha[front]), minlength=size)
    front_cover = 1.0 - np.exp(clear)
    covered = rgba[:, 3]
    partial = (front_cover > VISIBLE) & (front_cover < OPAQUE) & (covered > VISIBLE)
    shape = (deep.height, deep.width)
    return Flattened(rgba.reshape(*shape, 4), covered.reshape(shape), partial.reshape(shape)), cumulative


def flatten(deep: Deep) -> Flattened:
    """Owned front-to-back composites and partial-front masks; borrowed deep stays immutable and thread-safe.

    A pixel is partial when its front surface (samples within a small slab behind the nearest
    one) covers it only partly, so something behind it, or nothing, shows through.
    """
    rgba = np.empty((deep.height, deep.width, 4), np.float64)
    partial = np.empty((deep.height, deep.width), bool)
    offsets = deep.offsets
    cumulative = 0.0
    for row in range(deep.height):
        first, last = row * deep.width, (row + 1) * deep.width
        start, end = int(offsets[first]), int(offsets[last])
        values = Deep(
            deep.width,
            1,
            deep.counts[first:last],
            {name: channel[start:end] for name, channel in deep.channels.items()},
        )
        flattened, cumulative = _flatten_row(values, cumulative)
        rgba[row] = flattened.rgba[0]
        partial[row] = flattened.partial[0]
    return Flattened(rgba, rgba[..., 3], partial)


IDENTITY_CHANNELS = ("instance_id", "instance_check", "instance_check2", "coverage")


def instance_ids(deep: Deep, order: NDArray[np.int64] | None = None) -> NDArray[np.int64]:
    """Per-sample instance ids; `identity.UNLABELLED` where the identity check fails."""
    return identity.decode(*(deep.channels[name] for name in IDENTITY_CHANNELS), order=order)
