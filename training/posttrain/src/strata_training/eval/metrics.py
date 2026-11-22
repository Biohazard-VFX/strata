import numpy as np
from scipy import ndimage

from strata_training.types import Array


def mixed_errors(
    inverse_depth: Array,
    front_alpha: Array,
    target_inverse: Array,
    target_alpha: Array,
    mask: Array,
) -> tuple[Array, Array, Array]:
    indices = np.arange(target_alpha.shape[0], dtype=np.int64)[:, None, None]
    last = np.where(target_alpha > 1e-7, indices, 0).max(axis=0)[None]
    back = np.take_along_axis(inverse_depth, last, axis=0)[0]
    target_back = np.take_along_axis(target_inverse, last, axis=0)[0]
    return (
        np.abs(front_alpha[mask] - target_alpha[0][mask]),
        np.abs(inverse_depth[0][mask] / target_inverse[0][mask] - 1),
        np.abs(back[mask & (last[0] > 0)] / target_back[mask & (last[0] > 0)] - 1),
    )


def depth_edges(depth: Array, ratio: float) -> Array:
    log_depth = np.log(np.maximum(depth, 1e-8))
    edge = np.zeros(depth.shape, dtype=bool)
    edge[:, 1:] |= np.abs(np.diff(log_depth, axis=1)) > np.log(ratio)
    edge[1:, :] |= np.abs(np.diff(log_depth, axis=0)) > np.log(ratio)
    return edge


def boundary_recall(
    prediction: Array,
    truth: Array,
    ratio: float = 1.25,
    tolerance: int = 1,
) -> float | None:
    target = depth_edges(truth, ratio)
    if not target.any():
        return None
    predicted = depth_edges(prediction, ratio)
    if tolerance:
        predicted = ndimage.maximum_filter(predicted, size=2 * tolerance + 1)
    return float(predicted[target].mean())
