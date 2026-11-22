from collections.abc import Mapping

import numpy as np

from strata_training.data.targets import visible_inverse
from strata_training.types import Array, Layers


def oracle_metric_scale(data: Mapping[str, Array], inverse: Array) -> float:
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
