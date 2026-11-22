from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.io import atomic_path
from strata_training.types import Array, Layers

exr: Any = import_module("OpenEXR")


def read_rgb(path: Path, *, part_name: str = "C") -> Array:
    with exr.File(str(path), separate_channels=True) as file:
        candidates = [
            part
            for part in file.parts
            if part.name() == part_name or all(f"{part_name}.{c}" in part.channels for c in "RGB")
        ]
        if candidates:
            part = candidates[0]
        else:
            raise ValueError(f"Beauty is missing its {part_name} part: {path}")
        channels = part.channels
        return np.stack(
            [channels[c if part_name == "C" else f"{part_name}.{c}"].pixels for c in "RGB"]
        ).astype(np.float32)


def read_roles(path: Path) -> dict[str, Array]:
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        return {
            role: channels[f"roles.{role}"].pixels.astype(np.float32)
            for role in ("foreground", "hero", "fur")
        }


def write_deep(path: Path, prediction: Layers) -> None:
    height, width = prediction.alpha.shape[-2:]
    channels = {
        name: np.empty((height, width), dtype=object) for name in ("R", "G", "B", "A", "Z", "ZBack")
    }
    for y in range(height):
        for x in range(width):
            valid = (prediction.alpha[:, y, x] > 0) & (prediction.inverse_depth[:, y, x] > 0)
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
    with (
        atomic_path(path) as temporary,
        exr.File({"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels) as file,
    ):
        file.write(str(temporary))
