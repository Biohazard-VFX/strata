import numpy as np
from strata_contract.layers import reduce_samples as contract_reduce

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
    if alpha.shape != depth.shape or np.any(counts < 0):
        raise ValueError("alpha and depth shapes must match and counts must be nonnegative")
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError("deep colors and alpha must be finite")
    result = contract_reduce(
        depth.astype(np.float64),
        alpha.astype(np.float64),
        rgb.astype(np.float64),
        counts.astype(np.int64),
        shape,
        layers=layers,
        gap_ratio=gap_ratio,
    )
    return Layers(result.inverse_depth, result.alpha, result.rgb)


def reduce_identified_samples(
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: Array,
    shape: tuple[int, int],
    ids: Array,
    layers: int = 4,
) -> Layers:
    """Own reduced arrays; mask depth of any cluster containing visible sky, not its color/alpha."""
    result = reduce_samples(depth, alpha, rgb, counts, shape, layers)
    sky = np.repeat((alpha * (ids == 0))[:, None], 3, axis=1)
    membership = reduce_samples(depth, alpha, sky, counts, shape, layers)
    result.inverse_depth[membership.rgb[:, 0] > 1e-12] = 0
    return result
