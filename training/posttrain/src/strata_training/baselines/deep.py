import numpy as np
from scipy import ndimage

from strata_training.types import Array, Layers


def fake_deep(inverse: Array, beauty: Array, layers: int = 4) -> Layers:
    alpha = np.zeros((layers, *inverse.shape), dtype=np.float32)
    alpha[0] = 1
    rgb = np.zeros((layers, 3, *inverse.shape), dtype=np.float32)
    rgb[0] = beauty
    return Layers(np.repeat(inverse[None], layers, axis=0), alpha, rgb)


def analytic(inverse: Array, radius: int) -> tuple[Array, Array, Array]:
    size = radius * 2 + 1
    front = ndimage.percentile_filter(inverse, 95, size=size)
    back = ndimage.percentile_filter(inverse, 5, size=size)
    alpha = np.clip((inverse - back) / np.maximum(front - back, 1e-9), 0, 1)
    alpha[front / np.maximum(back, 1e-9) <= 1.15] = 1
    return alpha, front, back


def two_layer(alpha: Array, front: Array, back: Array, beauty: Array, layers: int = 4) -> Layers:
    result = fake_deep(back, beauty, layers)
    result.inverse_depth[0] = front
    result.alpha[0] = alpha
    result.alpha[1] = 1
    result.rgb[0] = beauty * alpha[None]
    result.rgb[1] = beauty
    return result
