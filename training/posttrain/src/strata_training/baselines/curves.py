import numpy as np

from strata_training.types import Array


def fit_curve(
    predictions: list[Array], targets: list[Array], masks: list[Array], bins: int = 20
) -> Array:
    sums = np.zeros(bins)
    counts = np.zeros(bins)
    for prediction, target, mask in zip(predictions, targets, masks, strict=True):
        index = np.minimum((prediction[mask] * bins).astype(int), bins - 1)
        counts += np.bincount(index, minlength=bins)
        sums += np.bincount(index, weights=target[mask], minlength=bins)
    return np.divide(sums, counts, out=(np.arange(bins) + 0.5) / bins, where=counts > 0)


def apply_curve(alpha: Array, curve: Array) -> Array:
    return curve[np.minimum((np.clip(alpha, 0, 1) * curve.size).astype(int), curve.size - 1)]
