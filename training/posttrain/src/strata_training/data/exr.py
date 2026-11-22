from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.data.targets import reduce_samples
from strata_training.io import atomic_path
from strata_training.types import Array, Layers

exr: Any = import_module("OpenEXR")


def read_rgb(path: Path, *, legacy: bool = False) -> Array:
    with exr.File(str(path), separate_channels=True) as file:
        candidates = [part for part in file.parts if part.name() == "C"]
        if candidates:
            part = candidates[0]
        elif legacy and len(file.parts) == 1:
            part = file.parts[0]
        else:
            raise ValueError(f"Beauty is missing its C part: {path}")
        channels = part.channels
        return np.stack([channels[c].pixels for c in "RGB"]).astype(np.float32)


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


def decode_instance_ids(premultiplied: Array, alpha: Array) -> Array:
    """Return owned rounded instance_id/A values, with zero for nonpositive alpha.

    Inputs are borrowed and unmodified; concurrent calls are safe when callers
    do not mutate inputs. The float array preserves read_roles decoding semantics.
    """
    if not np.all(np.isfinite(premultiplied) & (premultiplied >= 0)):
        raise ValueError("Deep instance IDs must be finite and nonnegative")
    return np.rint(np.divide(premultiplied, alpha, out=np.zeros_like(alpha), where=alpha > 0))


def read_roles(path: Path) -> dict[str, Array]:
    arrays, counts, shape = read_samples(path, ("Z", "A", "instance_id"))
    alpha, premultiplied = arrays["A"], arrays["instance_id"]
    ids = decode_instance_ids(premultiplied, alpha)
    markers = np.stack((ids == 3, (ids == 2) | (ids == 3), np.zeros_like(ids)), axis=1)
    coverage = reduce_samples(arrays["Z"], alpha, markers * alpha[:, None], counts, shape, 1)
    return {"fur": coverage.rgb[0, 0] > 1e-12, "foreground": coverage.rgb[0, 1] > 0.1}


def read_deep(path: Path, layers: int = 4) -> Layers:
    arrays, counts, shape = read_samples(path, "ZARGB")
    return reduce_samples(
        arrays["Z"],
        arrays["A"],
        np.stack([arrays[c] for c in "RGB"], axis=1),
        counts,
        shape,
        layers,
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
