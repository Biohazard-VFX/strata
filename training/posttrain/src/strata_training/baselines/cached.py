import json
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.baselines.curves import apply_curve
from strata_training.baselines.deep import fake_deep, two_layer
from strata_training.types import Array, Layers


def cached_baselines(identifier: str, root: Path, beauty: Array) -> dict[str, Layers]:
    data = root / identifier
    base = np.load(data / "base.npz")
    curves: dict[str, Any] = json.loads((root / "curves.json").read_text())["curves"]
    results = {
        "fake H+ oracle-aligned": fake_deep(base["inverse_depth"], beauty),
        "fake H+ raw metric": fake_deep(base["raw_inverse_depth"], beauty),
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
            for aligned in (False, True):
                ratio = 1.0 if aligned else float(base["ratio"])
                name = (
                    f"analytic r={radius}{suffix} {'oracle-aligned' if aligned else 'raw metric'}"
                )
                results[name] = two_layer(
                    alpha, arrays["inv_fg"] * ratio, arrays["inv_bg"] * ratio, beauty
                )
    probe = np.load(data / "probe.npz")
    results["v004 frozen probe oracle-aligned"] = two_layer(
        probe["alpha"], probe["inv_fg"], probe["inv_bg"], beauty
    )
    return results
