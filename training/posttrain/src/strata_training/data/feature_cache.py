from pathlib import Path

import numpy as np

from strata_training.data import mapped
from strata_training.types import Array

FEATURE_VERSION = "features-metric-v2"


def read_base(path: Path) -> dict[str, Array]:
    values = mapped.arrays(path)
    if (
        set(values) != {"version", "inverse_depth", "metric_scale"}
        or str(values["version"]) != FEATURE_VERSION
    ):
        raise ValueError("Feature cache version mismatch; extract into a new directory")
    return values


def base_arrays(inverse_depth: Array, metric_scale: float) -> dict[str, Array]:
    return {
        "version": np.array(FEATURE_VERSION),
        "inverse_depth": inverse_depth,
        "metric_scale": np.array(metric_scale),
    }
