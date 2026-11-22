"""Deep-pixel measurements: front-to-back flattening, partial-coverage and blended-id detection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from strata_datagen.exr import Deep

VISIBLE = 0.02
OPAQUE = 0.98
LAYER_GAP_M = 0.05
FRONT_SLAB = 0.05
ID_TOLERANCE = 0.05
ID_RELATIVE_TOLERANCE = 2e-5


@dataclass(frozen=True)
class Flattened:
    rgba: NDArray[np.float64]
    alpha: NDArray[np.float64]
    partial: NDArray[np.bool_]


def _segment_cumsum(
    values: NDArray[np.float64], starts: NDArray[np.int64], counts: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Exclusive cumulative sum within each pixel's run of samples."""
    total = np.cumsum(values) - values
    return total - np.repeat(total[starts[counts > 0]], counts[counts > 0])


def flatten(deep: Deep) -> Flattened:
    """Composite every pixel's samples front to back.

    A pixel is partial when its front surface (samples within a small slab behind the nearest
    one) covers it only partly, so something behind it, or nothing, shows through.
    """
    pixel = deep.pixel_index()
    order = np.lexsort((deep.channels["Z"], pixel))
    pixel = pixel[order]
    alpha = np.clip(deep.channels["A"][order].astype(np.float64), 0.0, 0.999999)
    starts = deep.offsets[:-1]
    transmittance = np.exp(_segment_cumsum(np.log1p(-alpha), starts, deep.counts))
    size = deep.counts.size
    rgba = np.zeros((size, 4))
    for i, name in enumerate("RGBA"):
        np.add.at(rgba[:, i], pixel, deep.channels[name][order] * transmittance)
    depth = deep.channels["Z"][order].astype(np.float64)
    near = np.full(size, np.inf)
    np.minimum.at(near, pixel, depth)
    front = depth <= near[pixel] * (1.0 + FRONT_SLAB) + LAYER_GAP_M
    clear = np.zeros(size)
    np.add.at(clear, pixel[front], np.log1p(-alpha[front]))
    front_cover = 1.0 - np.exp(clear)
    covered = rgba[:, 3]
    partial = (front_cover > VISIBLE) & (front_cover < OPAQUE) & (covered > VISIBLE)
    shape = (deep.height, deep.width)
    return Flattened(rgba.reshape(*shape, 4), covered.reshape(shape), partial.reshape(shape))


def _quotients(deep: Deep) -> NDArray[np.float64]:
    coverage = np.maximum(deep.channels["coverage"].astype(np.float64), 1e-12)
    return deep.channels["instance_id"] / coverage


def instance_ids(deep: Deep) -> NDArray[np.int64]:
    """Per-sample instance ids: both channels are premultiplied by the sample's float alpha."""
    return np.rint(_quotients(deep)).astype(np.int64)


def blended_ids(deep: Deep) -> NDArray[np.bool_]:
    """Samples whose id is not one surface's integer id but a coverage-weighted mean of several.

    Karma's float accumulation leaves an unblended id within about 1e-5 of its value relative to it.
    """
    quotients = _quotients(deep)
    tolerance = np.maximum(ID_TOLERANCE, ID_RELATIVE_TOLERANCE * np.abs(quotients))
    return np.abs(quotients - np.rint(quotients)) > tolerance
