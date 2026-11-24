from typing import cast

import numpy as np
import torch
from strata_dataset.layers import reduce_samples as contract_reduce
from strata_dataset.roles import sky as sample_sky

from strata_training.types import Array, Layers


def composite[T: (Array, torch.Tensor)](alpha: T, rgb: T) -> T:
    """Own beauty from final K-first layers; borrow NumPy arrays or device tensors."""
    trans = alpha[0] * 0 + 1
    result = rgb[0] * 0
    for index in range(alpha.shape[0]):
        result = result + trans[..., None, :, :] * rgb[index]
        trans = trans * (1 - alpha[index])
    return cast(T, result)


def visible_coverage(layers: Layers) -> Array:
    trans = np.concatenate(
        (np.ones_like(layers.alpha[:1]), np.cumprod(1 - layers.alpha[:-1], axis=0))
    )
    return trans * layers.alpha


def visible_inverse(layers: Layers) -> Array:
    weight = visible_coverage(layers)
    return (weight * layers.inverse_depth).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-8)


def reduce_samples(
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: Array,
    shape: tuple[int, int],
    layers: int = 4,
    gap_ratio: float = 1.15,
    *,
    sky: Array | None = None,
) -> Layers:
    if layers < 1 or gap_ratio <= 1:
        raise ValueError("layers must be positive and gap_ratio must exceed one")
    if alpha.shape != depth.shape or np.any(counts < 0):
        raise ValueError("alpha and depth shapes must match and counts must be nonnegative")
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError("deep colors and alpha must be finite")
    result = contract_reduce(
        depth,
        alpha,
        rgb,
        counts.astype(np.int64, copy=False),
        shape,
        layers=layers,
        gap_ratio=gap_ratio,
        sky=sky,
    )
    return Layers(result.inverse_depth, result.alpha, result.rgb)


def reduce_role_samples(
    depth: Array,
    alpha: Array,
    rgb: Array,
    counts: Array,
    shape: tuple[int, int],
    roles: Array,
    layers: int = 4,
) -> Layers:
    """Own reduced arrays; mask depth of any cluster containing visible sky, not its color/alpha."""
    return reduce_samples(
        depth,
        alpha,
        rgb,
        counts,
        shape,
        layers,
        sky=sample_sky(alpha, roles[:, 0]),
    )
