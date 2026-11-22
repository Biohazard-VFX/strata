from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from strata_training.data.targets import visible_inverse
from strata_training.types import Array, Layers


def oracle_metric_scale(entry: dict[str, Any], data: Mapping[str, Array], inverse: Array) -> float:
    if entry.get("legacy_data"):
        with np.load(Path(entry["legacy_data"]) / "gt.npz") as legacy:
            solid = ndimage.binary_erosion(
                (~legacy["two"]) & (legacy["A_tot"] > 0.999) & (legacy["inv_blend"] > 0),
                iterations=4,
            )
            truth = legacy["inv_blend"]
    else:
        solid = (
            (data["alpha"][0] > 0.999)
            & (data["alpha"][1:].sum(axis=0) < 1e-6)
            & (data["inverse_depth"][0] > 0)
        )
        truth = data["inverse_depth"][0]
    if not solid.any():
        truth = visible_inverse(Layers(data["inverse_depth"], data["alpha"], data["rgb"]))
        solid = truth > 0
    scale = float(np.exp(np.median(np.log(np.maximum(inverse[solid], 1e-8) / truth[solid]))))
    return 1 / scale
