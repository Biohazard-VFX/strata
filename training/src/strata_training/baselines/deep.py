import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from strata_training.types import Array, Layers


def fake_deep(inverse: Array, beauty: Array, layers: int = 4) -> Layers:
    alpha = np.zeros((layers, *inverse.shape), dtype=np.float32)
    alpha[0] = 1
    rgb = np.zeros((layers, 3, *inverse.shape), dtype=np.float32)
    rgb[0] = beauty
    return Layers(np.repeat(inverse[None], layers, axis=0), alpha, rgb)


def analytic(inverse: Array, radius: int) -> tuple[Array, Array, Array]:
    size = radius * 2 + 1
    front = ndimage.percentile_filter(inverse, 95, size=size)
    back = ndimage.percentile_filter(inverse, 5, size=size)
    alpha = np.clip((inverse - back) / np.maximum(front - back, 1e-9), 0, 1)
    alpha[front / np.maximum(back, 1e-9) <= 1.15] = 1
    return alpha, front, back


def two_layer(alpha: Array, front: Array, back: Array, beauty: Array, layers: int = 4) -> Layers:
    result = fake_deep(back, beauty, layers)
    result.inverse_depth[0] = front
    result.alpha[0] = alpha
    result.alpha[1] = 1
    result.rgb[0] = beauty * alpha[None]
    result.rgb[1] = beauty
    return result


def legacy_baselines(entry: dict[str, Any], beauty: Array, results: Path) -> dict[str, Layers]:
    data = Path(entry["legacy_data"])
    dp = np.load(data / "dp2.npz")
    curves: dict[str, Any] = json.loads((data.parent / "curves.json").read_text())
    predictions = {
        "fake H+ oracle-aligned": fake_deep(dp["dp_inv"], beauty),
        "fake H+ raw metric": fake_deep(1 / np.maximum(dp["depth"], 1e-8), beauty),
    }
    for radius in (6, 16):
        base = np.load(data / f"analytic_r{radius}.npz")
        a, f, b = base["alpha"], base["inv_fg"], base["inv_bg"]
        for curved in (False, True):
            alpha = (
                np.asarray(curves[str(radius)]["lut"])[np.minimum((a * 20).astype(int), 19)]
                if curved
                else a
            )
            suffix = "+curve" if curved else ""
            predictions[f"analytic r={radius}{suffix} oracle-aligned"] = two_layer(
                alpha, f, b, beauty
            )
            predictions[f"analytic r={radius}{suffix} raw metric"] = two_layer(
                alpha, f * float(dp["scale"]), b * float(dp["scale"]), beauty
            )
    probe = np.load(results / f"{entry['shot']}_probe_B.npz")
    predictions["v004 frozen probe oracle-aligned"] = two_layer(
        probe["alpha"], probe["inv_fg"], probe["inv_bg"], beauty
    )
    return predictions


def collapse_two(prediction: Layers) -> Layers:
    inv, alpha, rgb = prediction.inverse_depth, prediction.alpha, prediction.rgb
    trans = np.concatenate((np.ones_like(alpha[:1]), np.cumprod(1 - alpha[:-1], axis=0)))
    weights = alpha * trans
    gap = np.log(np.maximum(inv[:-1], 1e-8) / np.maximum(inv[1:], 1e-8))
    gap = np.where((weights[:-1] > 0.003) & (weights[1:] > 0.003), gap, -np.inf)
    split = np.argmax(gap, axis=0) + 1
    has_split: Array = np.asarray(np.isfinite(gap).any(axis=0))
    positions: Array = np.arange(inv.shape[0], dtype=np.int64)[:, None, None]
    front: Array = np.logical_or(np.less(positions, np.asarray(split)), np.logical_not(has_split))
    front_weight = (weights * front).sum(axis=0)
    back_weight = (weights * ~front).sum(axis=0)
    first_inv = (weights * inv * front).sum(axis=0) / np.maximum(front_weight, 1e-8)
    last_inv = (weights * inv * ~front).sum(axis=0) / np.maximum(back_weight, 1e-8)
    last_inv = np.where(back_weight > 1e-8, last_inv, first_inv)
    first_rgb = (trans[:, None] * rgb * front[:, None]).sum(axis=0)
    last_rgb = (trans[:, None] * rgb * (~front)[:, None]).sum(axis=0) / np.maximum(
        1 - front_weight, 1e-8
    )[None]
    back_alpha = back_weight / np.maximum(1 - front_weight, 1e-8)
    return Layers(
        np.stack((first_inv, last_inv)),
        np.stack((front_weight, back_alpha)),
        np.stack((first_rgb, last_rgb)),
    )
