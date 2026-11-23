from typing import Any

import numpy as np
from scipy import ndimage

from strata_training.types import Array, Layers


def leakage_samples(
    prediction: Layers, target: Layers, mixed: Array, beauty: Array, region: Array | None = None
) -> dict[str, Array]:
    inverse = target.inverse_depth[0].astype(np.float64)
    depth = np.divide(1.0, inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
    plane = depth + np.maximum(0.01, 0.01 * depth)
    before = (prediction.inverse_depth > 0) & (prediction.inverse_depth >= 1 / plane)
    transmitted = np.prod(np.where(before, 1 - prediction.alpha.astype(np.float64), 1), axis=0)
    coverage = 1 - np.prod(1 - target.alpha.astype(np.float64), axis=0)
    foreground = target.alpha[0] > 0.5
    boundary = ndimage.binary_dilation(foreground) ^ ndimage.binary_erosion(
        foreground, border_value=1
    )
    edges = boundary | mixed.astype(bool)
    away = ndimage.distance_transform_edt(~edges) > 4 if edges.any() else ~edges
    interior = (coverage >= 0.999) & (inverse > 0) & away
    if region is not None:
        interior &= region
    brightness = beauty.max(axis=0)
    masks = {
        "all": interior,
        "bright": interior & (brightness > 1.4),
        "normal": interior & (brightness > 0.04) & (brightness <= 1.4),
        "dark": interior & (brightness <= 0.04),
    }
    return {name: transmitted[mask] for name, mask in masks.items()}


def leakage_summary(samples: dict[str, Array]) -> dict[str, Any]:
    regions = {
        name: {"mean": float(values.mean()) if values.size else None, "pixels": int(values.size)}
        for name, values in samples.items()
    }
    means = [float(row["mean"]) for row in regions.values() if row["mean"] is not None]
    worst = max(means) if means else None
    return {
        **regions,
        "worst_mean": worst,
        "passed": worst is not None and np.isfinite(worst) and worst <= 0.001,
    }


def validation_improves(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    leak, previous = candidate["leakage"], best["leakage"]
    if leak["passed"] != previous["passed"]:
        return bool(leak["passed"])
    mixed, old_mixed = candidate["mixed_alpha_mae"], best["mixed_alpha_mae"]
    score, old_score = leak["worst_mean"], previous["worst_mean"]
    return mixed <= old_mixed and score <= old_score and (mixed < old_mixed or score < old_score)
