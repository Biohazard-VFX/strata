from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.data.targets import reduce_samples
from strata_training.types import Array, Layers

exr: Any = import_module("OpenEXR")


def read_rgb(path: Path) -> Array:
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        return np.stack([channels[c].pixels for c in "RGB"]).astype(np.float32)


def read_deep(path: Path, layers: int = 4) -> Layers:
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        z: Array = channels["Z"].pixels
        counts = np.fromiter((len(v) if v is not None else 0 for v in z.flat), np.int64)

        def flatten(name: str) -> Array:
            return np.concatenate([v for v in channels[name].pixels.flat if v is not None]).astype(
                np.float64
            )

        return reduce_samples(
            flatten("Z"),
            flatten("A"),
            np.stack([flatten(c) for c in "RGB"], axis=1),
            counts,
            z.shape,
            layers,
        )


def write_deep(path: Path, prediction: Layers, cutoff: float = 1e-4) -> None:
    height, width = prediction.alpha.shape[-2:]
    channels = {
        name: np.empty((height, width), dtype=object) for name in ("R", "G", "B", "A", "Z", "ZBack")
    }
    for y in range(height):
        for x in range(width):
            valid = (prediction.alpha[:, y, x] > cutoff) & (prediction.inverse_depth[:, y, x] > 0)
            depth = 1 / prediction.inverse_depth[valid, y, x]
            order = np.argsort(depth, kind="stable")
            values = {
                "A": prediction.alpha[valid, y, x][order],
                "Z": depth[order],
                "ZBack": depth[order],
            }
            values.update({c: prediction.rgb[valid, i, y, x][order] for i, c in enumerate("RGB")})
            for name in channels:
                channels[name][y, x] = values[name].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exr.File(
        {"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels
    ) as file:
        file.write(str(path))
