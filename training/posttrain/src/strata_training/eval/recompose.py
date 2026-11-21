import numpy as np

from strata_training.types import Array, Layers


def insert_plane(layers: Layers, depth: float = 3.0) -> Array:
    alpha = layers.alpha * (layers.inverse_depth > 1 / depth)
    rgb = layers.rgb * (layers.inverse_depth > 1 / depth)[:, None]
    trans = np.concatenate((np.ones_like(alpha[:1]), np.cumprod(1 - alpha[:-1], axis=0)))
    result = (trans[:, None] * rgb).sum(axis=0)
    return result + np.prod(1 - alpha, axis=0)[None] * np.array([1, 0, 1])[:, None, None]
