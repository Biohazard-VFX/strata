import numpy as np

from strata_training.types import Array, Layers


def composite(alpha: Array, rgb: Array) -> Array:
    trans = np.concatenate((np.ones_like(alpha[:1]), np.cumprod(1 - alpha[:-1], axis=0)))
    return (trans[:, None] * rgb).sum(axis=0)


def visible_inverse(layers: Layers) -> Array:
    trans = np.concatenate(
        (np.ones_like(layers.alpha[:1]), np.cumprod(1 - layers.alpha[:-1], axis=0))
    )
    weight = trans * layers.alpha
    return (weight * layers.inverse_depth).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-8)


def reduce_samples(
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: Array,
    shape: tuple[int, int],
    layers: int = 4,
    gap_ratio: float = 1.15,
) -> Layers:
    if layers < 1 or gap_ratio <= 1:
        raise ValueError("layers must be positive and gap_ratio must exceed one")
    pixels = shape[0] * shape[1]
    if counts.size != pixels or int(counts.sum()) != depth.size or rgb.shape != (depth.size, 3):
        raise ValueError("deep channel sizes must agree with sample counts and image shape")
    if alpha.shape != depth.shape or np.any(counts < 0):
        raise ValueError("alpha and depth shapes must match and counts must be nonnegative")
    if not (np.isfinite(depth).all() and np.all(depth > 0)):
        raise ValueError("sample depths must be finite and positive")
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError("deep colors and alpha must be finite")
    output = np.zeros((pixels * layers, 5), dtype=np.float64)
    if not depth.size:
        return _reshape(output, shape, layers)
    pix = np.repeat(np.arange(pixels), counts)
    order = np.lexsort((depth, pix))
    z, a, c = depth[order], np.clip(alpha[order], 0, 1), rgb[order]
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    log_t = np.log1p(-np.minimum(a, 1 - 1e-15))
    cumulative = np.cumsum(log_t, dtype=np.float64)
    before = np.concatenate(([0.0], cumulative[:-1]))
    trans = np.exp(before - before[starts])
    weight = trans * a
    visible = weight > 1e-12
    z, a, c = z[visible], a[visible], c[visible]
    pix, trans, weight = pix[visible], trans[visible], weight[visible]
    if not z.size:
        return _reshape(output, shape, layers)
    counts = np.bincount(pix, minlength=pixels)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    gaps = np.full(z.size, -np.inf)
    same = pix[1:] == pix[:-1]
    gaps[1:] = np.where(same, np.diff(np.log(z)), -np.inf)
    pixel_starts = np.flatnonzero(np.concatenate(([True], ~same)))
    pixel_counts = counts[counts > 0]
    split = np.zeros(z.size, dtype=bool)
    for _ in range(layers - 1):
        maxima = np.maximum.reduceat(gaps, pixel_starts)
        candidates = np.flatnonzero(
            (gaps == np.repeat(maxima, pixel_counts)) & (gaps > np.log(gap_ratio))
        )
        candidate_pixels = pix[candidates]
        chosen = candidates[np.diff(candidate_pixels, prepend=-1) != 0]
        split[chosen] = True
        gaps[chosen] = -np.inf
    number = np.cumsum(split)
    index = pix * layers + number - number[starts]
    visible = np.bincount(index, weights=weight, minlength=pixels * layers)
    trans_start = np.zeros(pixels * layers)
    cluster_starts = np.flatnonzero(np.concatenate(([True], index[1:] != index[:-1])))
    trans_start[index[cluster_starts]] = np.maximum.reduceat(trans, cluster_starts)
    local_trans = np.divide(
        trans, trans_start[index], out=np.zeros_like(trans), where=trans_start[index] > 1e-12
    )
    output[:, 0] = np.divide(
        np.bincount(index, weights=weight / z, minlength=pixels * layers),
        visible,
        out=np.zeros_like(visible),
        where=visible > 1e-12,
    )
    output[:, 1] = np.divide(
        visible, trans_start, out=np.zeros_like(visible), where=trans_start > 1e-12
    )
    for channel in range(3):
        output[:, channel + 2] = np.bincount(
            index, weights=local_trans * c[:, channel], minlength=pixels * layers
        )
    output[visible <= 1e-12] = 0
    return _reshape(output, shape, layers)


def _reshape(output: Array, shape: tuple[int, int], layers: int) -> Layers:
    cube = output.reshape(*shape, layers, 5).transpose(2, 3, 0, 1).astype(np.float32)
    return Layers(cube[:, 0], np.clip(cube[:, 1], 0, 1), cube[:, 2:])
