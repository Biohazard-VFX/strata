from math import fsum
from typing import Any

import numpy as np

from strata_training.eval.coverage import coverage_regions
from strata_training.types import Array, Layers


def leakage_samples(
    prediction: Layers, target: Layers, mixed: Array, beauty: Array, region: Array | None = None
) -> dict[str, Array]:
    inverse = target.inverse_depth[0].astype(np.float64)
    depth = np.divide(1.0, inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
    offset = np.maximum(0.01, 0.01 * depth)
    plane = depth + offset
    before = (prediction.inverse_depth > 0) & (prediction.inverse_depth >= 1 / plane)
    transmitted = np.prod(np.where(before, 1 - prediction.alpha.astype(np.float64), 1), axis=0)
    _, interior = coverage_regions(target, mixed)
    interior &= inverse > 0
    if region is not None:
        interior &= region
    brightness = beauty.max(axis=0)
    masks = {
        "all": interior,
        "bright": interior & (brightness > 1.4),
        "normal": interior & (brightness > 0.04) & (brightness <= 1.4),
        "dark": interior & (brightness <= 0.04),
    }
    premature_plane = np.subtract(depth, offset, out=np.zeros_like(depth), where=inverse > 0)
    premature_before = (prediction.inverse_depth > 0) & (
        prediction.inverse_depth >= 1 / np.maximum(premature_plane, np.finfo(np.float64).tiny)
    )
    premature = 1 - np.prod(
        np.where(premature_before, 1 - prediction.alpha.astype(np.float64), 1), axis=0
    )
    return {
        **{name: transmitted[mask] for name, mask in masks.items()},
        "premature": premature[interior & (premature_plane > 0)],
    }


def leakage_summary(samples: dict[str, Array]) -> dict[str, Any]:
    regions: dict[str, Any] = {}
    for name, values in samples.items():
        count = int(values.size)
        mean = fsum(values) / count if count else None
        evaluated = count > 0 if name == "all" else count >= 250
        limit = 0.001 if name == "all" else max(0.001, 25 / count) if count else None
        regions[name] = {
            "mean": mean,
            "pixels": count,
            "limit": limit if name != "premature" else None,
            "status": (
                "not evaluated"
                if not evaluated or name == "premature"
                else "pass"
                if mean is not None and limit is not None and np.isfinite(mean) and mean <= limit
                else "fail"
            ),
        }
    means = [
        float(row["mean"])
        for name, row in regions.items()
        if name != "premature" and row["status"] != "not evaluated"
    ]
    worst = max(means) if means else None
    return {
        **regions,
        "worst_mean": worst,
        "passed": regions["all"]["status"] == "pass"
        and all(row["status"] != "fail" for row in regions.values()),
    }
