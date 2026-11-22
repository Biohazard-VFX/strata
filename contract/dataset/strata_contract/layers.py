"""K-layer depth targets from deep samples: the merge rule shared by data generation and training.

Samples are sorted by depth per pixel; each pixel splits at its largest K-1 gaps in log depth that
exceed log(gap_ratio); samples between splits merge front to back. Per layer the result is the
coverage-weighted inverse depth, the layer's own alpha (coverage over the transmittance reaching
it) and its premultiplied colour composited within the layer.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
LAYERS = 4
GAP_RATIO = 1.15


@dataclass(frozen=True)
class Layers:
    """inverse_depth and alpha are (K, H, W); rgb is (K, 3, H, W)."""

    inverse_depth: NDArray[np.float32]
    alpha: NDArray[np.float32]
    rgb: NDArray[np.float32]


def _split(z: Array, pix: NDArray[np.int64], pixels: int, layers: int, gap_ratio: float) -> NDArray[np.bool_]:
    gaps = np.full(z.size, -np.inf)
    gaps[1:] = np.where(pix[1:] == pix[:-1], np.diff(np.log(z)), -np.inf)
    split = np.zeros(z.size, dtype=bool)
    for _ in range(layers - 1):
        maxima = np.full(pixels, -np.inf)
        np.maximum.at(maxima, pix, gaps)
        candidates = (gaps == maxima[pix]) & (gaps > np.log(gap_ratio))
        first = np.full(pixels, z.size)
        np.minimum.at(first, pix[candidates], np.flatnonzero(candidates))
        chosen = first[first < z.size]
        split[chosen] = True
        gaps[chosen] = -np.inf
    return split


def reduce_samples(
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: NDArray[np.int64],
    shape: tuple[int, int],
    *,
    layers: int = LAYERS,
    gap_ratio: float = GAP_RATIO,
) -> Layers:
    """Reduce per-pixel deep samples (pixel order, any depth order; rgb premultiplied) to K layers."""
    pixels = shape[0] * shape[1]
    if counts.size != pixels or int(counts.sum()) != depth.size or rgb.shape != (depth.size, 3):
        raise ValueError("deep channel sizes must agree with sample counts and image shape")
    if not (np.isfinite(depth).all() and np.all(depth > 0)):
        raise ValueError("sample depths must be finite and positive")
    output = np.zeros((pixels * layers, 5), dtype=np.float64)
    if depth.size:
        _fill(output, depth, alpha, rgb, counts, pixels=pixels, layers=layers, gap_ratio=gap_ratio)
    cube = output.reshape(*shape, layers, 5).transpose(2, 3, 0, 1).astype(np.float32)
    return Layers(cube[:, 0], np.clip(cube[:, 1], 0, 1), cube[:, 2:])


def _fill(
    output: Array,
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: NDArray[np.int64],
    *,
    pixels: int,
    layers: int,
    gap_ratio: float,
) -> None:
    pix: NDArray[np.int64] = np.repeat(np.arange(pixels, dtype=np.int64), counts)
    order = np.lexsort((depth, pix))
    z, a, c = depth[order], np.clip(alpha[order], 0, 1), rgb[order]
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    cumulative = np.cumsum(np.log1p(-np.minimum(a, 1 - 1e-15)), dtype=np.float64)
    before = np.concatenate(([0.0], cumulative[:-1]))
    trans = np.exp(before - before[starts])
    weight = trans * a
    keep = weight > 1e-12
    z, c, pix, trans, weight = z[keep], c[keep], pix[keep], trans[keep], weight[keep]
    if not z.size:
        return
    counts = np.bincount(pix, minlength=pixels)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    number = np.cumsum(_split(z, pix, pixels, layers, gap_ratio))
    index = pix * layers + number - number[starts]
    size = pixels * layers
    visible = np.bincount(index, weights=weight, minlength=size)
    trans_start = np.zeros(size)
    np.maximum.at(trans_start, index, trans)
    local = np.divide(trans, trans_start[index], out=np.zeros_like(trans), where=trans_start[index] > 1e-12)
    output[:, 0] = np.divide(
        np.bincount(index, weights=weight / z, minlength=size),
        visible,
        out=np.zeros_like(visible),
        where=visible > 1e-12,
    )
    output[:, 1] = np.divide(visible, trans_start, out=np.zeros_like(visible), where=trans_start > 1e-12)
    for channel in range(3):
        output[:, channel + 2] = np.bincount(index, weights=local * c[:, channel], minlength=size)
    output[visible <= 1e-12] = 0
