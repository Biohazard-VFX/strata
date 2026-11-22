from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.data.targets import reduce_samples
from strata_training.io import atomic_path
from strata_training.types import Array, Layers

exr: Any = import_module("OpenEXR")


def read_rgb(path: Path, *, legacy: bool = False, part_name: str = "C") -> Array:
    with exr.File(str(path), separate_channels=True) as file:
        candidates = [
            part
            for part in file.parts
            if part.name() == part_name or all(f"{part_name}.{c}" in part.channels for c in "RGB")
        ]
        if candidates:
            part = candidates[0]
        elif legacy and part_name == "C" and len(file.parts) == 1:
            part = file.parts[0]
        else:
            raise ValueError(f"Beauty is missing its {part_name} part: {path}")
        channels = part.channels
        return np.stack(
            [channels[c if part_name == "C" else f"{part_name}.{c}"].pixels for c in "RGB"]
        ).astype(np.float32)


def read_samples(
    path: Path, names: str | tuple[str, ...] = ("Z", "A", "R", "G", "B", "instance_id")
) -> tuple[dict[str, Array], Array, tuple[int, int]]:
    """Return channel arrays, row-major pixel counts and (height, width).

    Arrays are owned float64 copies in original per-pixel sample order, valid
    after file close. Counts are owned int64 values, including empty pixels.
    Channels retain stored values, including premultiplied RGB and instance_id;
    request ZBack explicitly when needed. Missing requested channels raise KeyError.
    Calls share no mutable state; callers own synchronization of returned arrays.
    """
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        z: Array = channels["Z"].pixels
        counts = np.fromiter((len(v) if v is not None else 0 for v in z.flat), np.int64)
        arrays: dict[str, Array] = {}
        for name in names:
            chunks: list[Array] = [v for v in channels[name].pixels.flat if v is not None]
            arrays[name] = (
                np.concatenate(chunks).astype(np.float64)
                if chunks
                else np.empty(0, dtype=np.float64)
            )
        return arrays, counts, z.shape


def read_roles(path: Path) -> dict[str, Array]:
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        return {
            role: channels[f"roles.{role}"].pixels.astype(np.float32)
            for role in ("foreground", "hero", "fur")
        }


def read_deep(path: Path, layers: int = 4, *, background: bool = False) -> Layers:
    from strata_dataset.roles import nested, sky

    if not background:
        arrays, counts, shape = read_samples(path, "ZARGB")
        return reduce_samples(
            arrays["Z"],
            arrays["A"],
            np.stack([arrays[c] for c in "RGB"], axis=1),
            counts,
            shape,
            layers,
        )
    arrays, counts, shape = read_samples(
        path, ("Z", "coverage", "R", "G", "B", "foreground", "hero", "fur")
    )
    alpha = np.clip(arrays["coverage"], 0, 1).astype(np.float32)
    weights = nested(np.stack([arrays[c] for c in ("foreground", "hero", "fur")], axis=1), alpha)
    return reduce_samples(
        arrays["Z"],
        alpha,
        np.stack([arrays[c] for c in "RGB"], axis=1),
        counts,
        shape,
        layers,
        sky=sky(alpha, weights[:, 0]) if background else None,
    )


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
