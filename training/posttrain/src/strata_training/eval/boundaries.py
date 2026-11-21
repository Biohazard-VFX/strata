import numpy as np
from scipy import ndimage

from strata_training.types import Array

THRESHOLDS = (5, 10, 15, 20, 25)


def _ratios(inverse: Array) -> tuple[Array, Array, Array, Array]:
    inverse = np.maximum(inverse, 1e-8)
    right = inverse[:, :-1] / inverse[:, 1:]
    down = inverse[:-1] / inverse[1:]
    return right, 1 / right, down, 1 / down


def _nms(ratio: Array, threshold: float, axis: int) -> Array:
    active = ratio > threshold
    structure = np.zeros((3, 3), dtype=bool)
    if axis == 1:
        structure[1, :] = True
    else:
        structure[:, 1] = True
    labels, number = ndimage.label(active, structure)
    if not number:
        return active
    maxima = ndimage.maximum(ratio, labels, np.arange(1, number + 1))
    return active & (ratio >= np.concatenate(([np.inf], maxima))[labels])


def directed_boundary_counts(
    prediction: Array, truth: Array, region: Array | None = None
) -> list[tuple[int, int, int]]:
    predicted, target = _ratios(prediction), _ratios(truth)
    counts: list[tuple[int, int, int]] = []
    for threshold in THRESHOLDS:
        tp, predicted_count, target_count = 0, 0, 0
        for i, (p, t) in enumerate(zip(predicted, target, strict=True)):
            pred, gt = p > 1 + threshold / 100, t > 1 + threshold / 100
            valid = truth > 0
            known = (valid[:, :-1] & valid[:, 1:]) if i < 2 else (valid[:-1] & valid[1:])
            pred, gt = pred & known, gt & known
            if region is not None:
                mask = (region[:, :-1] | region[:, 1:]) if i < 2 else (region[:-1] | region[1:])
                pred, gt = pred & mask, gt & mask
            tp += int((pred & gt).sum())
            predicted_count += int(pred.sum())
            target_count += int(gt.sum())
        counts.append((tp, predicted_count, target_count))
    return counts


def boundary_f1(counts: list[tuple[int, int, int]]) -> float | None:
    scores: list[float] = []
    weights: list[int] = []
    for threshold, (tp, predicted, target) in zip(THRESHOLDS, counts, strict=True):
        if target:
            scores.append(2 * tp / max(predicted + target, 1))
            weights.append(threshold)
    return float(np.average(scores, weights=weights)) if scores else None


def silhouette_recall(
    inverse: Array, silhouette: Array, region: Array | None = None
) -> float | None:
    b = silhouette.astype(bool)
    masks = (b[:, :-1] & ~b[:, 1:], b[:, 1:] & ~b[:, :-1], b[:-1] & ~b[1:], b[1:] & ~b[:-1])
    if region is not None:
        allowed = (
            region[:, :-1] | region[:, 1:],
            region[:, :-1] | region[:, 1:],
            region[:-1] | region[1:],
            region[:-1] | region[1:],
        )
        masks = tuple(m & a for m, a in zip(masks, allowed, strict=True))
    denominator = sum(int(m.sum()) for m in masks)
    if not denominator:
        return None
    values: list[float] = []
    for threshold in THRESHOLDS:
        hits = 0
        for i, (ratio, mask) in enumerate(zip(_ratios(inverse), masks, strict=True)):
            hits += int((_nms(ratio, 1 + threshold / 100, 1 if i < 2 else 0) & mask).sum())
        values.append(hits / denominator)
    return float(np.average(values, weights=THRESHOLDS))
