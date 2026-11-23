import json
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.baselines.curves import apply_curve
from strata_training.baselines.deep import fake_deep, two_layer
from strata_training.data.feature_cache import read_base
from strata_training.types import Array, Layers


def cached_baselines(identifier: str, root: Path, beauty: Array) -> dict[str, Layers]:
    data = root / identifier
    base = read_base(data / "base.npz")
    curves: dict[str, Any] = json.loads((root / "curves.json").read_text())["curves"]
    results = {
        "fake H+ raw metric": fake_deep(base["inverse_depth"], beauty),
    }
    for radius in (6, 16):
        arrays = np.load(data / f"r{radius}.npz")
        for curved in (False, True):
            alpha = (
                apply_curve(arrays["alpha"], np.asarray(curves[str(radius)]))
                if curved
                else arrays["alpha"]
            )
            suffix = "+curve" if curved else ""
            results[f"analytic r={radius}{suffix} raw metric"] = two_layer(
                alpha, arrays["inv_fg"], arrays["inv_bg"], beauty
            )
    probe = np.load(data / "probe.npz")
    results["frozen probe raw metric"] = two_layer(
        probe["alpha"], probe["inv_fg"], probe["inv_bg"], beauty
    )
    return results
