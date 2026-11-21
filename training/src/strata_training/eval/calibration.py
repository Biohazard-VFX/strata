from typing import TypedDict

import numpy as np

from strata_training.types import Array


class Calibration(TypedDict):
    ece: float
    brier: float
    counts: list[int]
    predicted: list[float]
    observed: list[float]


def calibration(alpha: Array, truth: Array, bins: int = 10) -> Calibration:
    index = np.minimum((np.clip(alpha, 0, 1) * bins).astype(int), bins - 1)
    count = np.bincount(index, minlength=bins)
    predicted = np.bincount(index, weights=alpha, minlength=bins)
    actual = np.bincount(index, weights=truth, minlength=bins)
    means = np.divide(predicted, count, out=np.zeros(bins), where=count > 0)
    targets = np.divide(actual, count, out=np.zeros(bins), where=count > 0)
    return {
        "ece": float((np.abs(means - targets) * count).sum() / max(int(count.sum()), 1)),
        "brier": float(np.mean((alpha - truth) ** 2)),
        "counts": count.tolist(),
        "predicted": means.tolist(),
        "observed": targets.tolist(),
    }


def tracked_depth_jitter(predicted_z: Array, true_z: Array, visible: Array) -> float | None:
    valid = (
        visible[1:]
        & visible[:-1]
        & (predicted_z[1:] > 0)
        & (predicted_z[:-1] > 0)
        & (true_z[1:] > 0)
        & (true_z[:-1] > 0)
    )
    if not valid.any():
        return None
    residual = np.log(np.maximum(predicted_z, 1e-8)) - np.log(np.maximum(true_z, 1e-8))
    return float(np.abs(np.diff(residual, axis=0))[valid].mean())
