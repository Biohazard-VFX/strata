from itertools import pairwise
from typing import Any

import numpy as np
from scipy import ndimage

from strata_training.types import Array, Layers

FAR_DEPTH = 200.0


def coverage_integrals(front: Array, alpha: Array, z: Array, predicted: Array) -> Array:
    front, z = (np.minimum(value.astype(np.float64), FAR_DEPTH) for value in (front, z))
    alpha, predicted = alpha.astype(np.float64), predicted.astype(np.float64)
    if np.any(front <= 0) or np.any(z <= 0):
        raise ValueError("Coverage sweep requires positive depths")
    endpoints = np.sort(np.concatenate((front, z, np.full_like(z[:1], FAR_DEPTH))), axis=0)
    result = np.zeros((2, *front.shape[1:]), dtype=np.float64)
    for low, high in pairwise(endpoints):
        middle = np.sqrt(low * high)
        gt_t = np.prod(np.where(front <= middle[None], 1 - alpha, 1), axis=0)
        pred_t = np.prod(np.where(z <= middle[None], 1 - predicted, 1), axis=0)
        delta = gt_t - pred_t
        width = np.log(high / low)
        result[0] += np.maximum(delta, 0) * width
        result[1] += np.maximum(-delta, 0) * width
    return result


def coverage_samples(prediction: Layers, target: Layers, mixed: Array) -> dict[str, Array]:
    def depths(layers: Layers) -> Array:
        if not np.isfinite(layers.inverse_depth).all() or np.any(layers.inverse_depth < 0):
            raise FloatingPointError("Invalid coverage-sweep inverse depth")
        return np.divide(
            1.0,
            layers.inverse_depth,
            out=np.full_like(layers.inverse_depth, FAR_DEPTH, dtype=np.float64),
            where=layers.inverse_depth > 0,
        )

    gt_z, pred_z = depths(target), depths(prediction)
    output = np.empty((2, *mixed.shape), dtype=np.float64)
    for row in range(0, mixed.shape[0], 32):
        part = slice(row, row + 32)
        output[:, part] = coverage_integrals(
            gt_z[:, part],
            target.alpha[:, part],
            pred_z[:, part],
            prediction.alpha[:, part],
        )
    band, interior = coverage_regions(target, mixed)
    return {
        f"{region}_{direction}": output[index][mask]
        for region, mask in (("edge", band), ("interior", interior))
        for index, direction in enumerate(("too_near", "too_far"))
    }


def coverage_regions(target: Layers, mixed: Array) -> tuple[Array, Array]:
    foreground = target.alpha[0] > 0.5
    boundary = ndimage.binary_dilation(foreground) ^ ndimage.binary_erosion(
        foreground, border_value=1
    )
    edges = boundary | mixed.astype(bool)
    band = ndimage.distance_transform_edt(~edges) <= 4 if edges.any() else edges
    interior = (1 - np.prod(1 - target.alpha.astype(np.float64), axis=0) >= 0.999) & ~band
    return band, interior


def coverage_summary(samples: dict[str, Array]) -> dict[str, Any]:
    regions: dict[str, Any] = {}
    for region in ("edge", "interior"):
        near, far = (samples[f"{region}_{direction}"] for direction in ("too_near", "too_far"))
        count = int(near.size)
        regions[region] = {
            "pixels": count,
            "too_near": float(near.mean()) if count else None,
            "too_far": float(far.mean()) if count else None,
            "mean": float((near + far).mean()) if count else None,
        }
    means = [regions[region]["mean"] for region in ("edge", "interior")]
    return {
        "reference": "layer-targets",
        **regions,
        "selection": float(np.mean(means)) if all(value is not None for value in means) else None,
    }
