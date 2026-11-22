"""Raw-sample sidecars: the per-frame memory-mappable deep samples and images of a recomposition frame.

A frame directory holds `noisy.npy` and `denoised.npy` (float32, 3 x H x W) and `deep/` with
`shape.npy` (int64 (2,)), `offsets.npy` (int64 (H*W+1,)), `z.npy`, `alpha.npy` (float32 (N,)),
`rgb.npy` (float32 (N, 3), premultiplied) and `ids.npy` (int64 (N,)): row-major pixels, samples
front to back within a pixel. Arrays are uncompressed little-endian C-order NPY, no pickles.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEEP_FIELDS: dict[str, str] = {
    "shape": "<i8",
    "offsets": "<i8",
    "z": "<f4",
    "alpha": "<f4",
    "rgb": "<f4",
    "ids": "<i8",
}
IMAGES = ("noisy", "denoised")


def frame_id(shot_id: str, frame: int) -> str:
    return f"{shot_id}_{frame:04d}"


def load(directory: Path) -> dict[str, NDArray[np.generic]]:
    """Every array of a frame directory, keyed by file stem (deep fields unprefixed)."""
    arrays = {name: np.load(directory / f"{name}.npy", allow_pickle=False) for name in IMAGES}
    arrays.update(
        {name: np.load(directory / "deep" / f"{name}.npy", allow_pickle=False) for name in DEEP_FIELDS}
    )
    return arrays


def _layout_errors(arrays: dict[str, NDArray[np.generic]], shape: tuple[int, int]) -> list[str]:
    errors = [
        f"{name}: dtype {arrays[name].dtype.str}, expected {dtype}"
        for name, dtype in DEEP_FIELDS.items()
        if arrays[name].dtype.str != dtype
    ]
    errors += [
        f"{name}: dtype {arrays[name].dtype.str}, expected <f4"
        for name in IMAGES
        if arrays[name].dtype.str != "<f4"
    ]
    errors += [f"{name}: not C-contiguous" for name, a in arrays.items() if not a.flags.c_contiguous]
    if arrays["shape"].tolist() != list(shape):
        errors.append(f"shape {arrays['shape'].tolist()} differs from the frame {list(shape)}")
    errors += [f"{name}: shape {arrays[name].shape}" for name in IMAGES if arrays[name].shape != (3, *shape)]
    return errors


def _sample_errors(arrays: dict[str, NDArray[np.generic]], pixels: int) -> list[str]:
    offsets = arrays["offsets"].astype(np.int64)
    if offsets.shape != (pixels + 1,) or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        return ["offsets must start at 0, be nondecreasing and have H*W+1 entries"]
    count = int(offsets[-1])
    expected = {"z": (count,), "alpha": (count,), "rgb": (count, 3), "ids": (count,)}
    errors = [
        f"{name}: shape {arrays[name].shape}, expected {s}"
        for name, s in expected.items()
        if arrays[name].shape != s
    ]
    if errors:
        return errors
    z, alpha = arrays["z"].astype(np.float64), arrays["alpha"].astype(np.float64)
    if not (np.isfinite(z).all() and np.all(z > 0)):
        errors.append("z must be finite and positive")
    if not (np.isfinite(alpha).all() and np.all((alpha >= 0) & (alpha <= 1))):
        errors.append("alpha must be in [0, 1]")
    if not np.isfinite(arrays["rgb"]).all() or not all(np.isfinite(arrays[n]).all() for n in IMAGES):
        errors.append("rgb and images must be finite")
    if np.any(arrays["ids"].astype(np.int64) < 0):
        errors.append("ids must be nonnegative")
    pixel = np.repeat(np.arange(pixels), np.diff(offsets))
    if np.any((pixel[1:] == pixel[:-1]) & (z[1:] < z[:-1])):
        errors.append("samples must be front to back within each pixel")
    return errors


def errors(directory: Path, shape: tuple[int, int]) -> list[str]:
    """Contract violations of one frame directory for an (H, W) frame; empty when it conforms."""
    missing = [f"{n}.npy" for n in IMAGES if not (directory / f"{n}.npy").is_file()]
    missing += [f"deep/{n}.npy" for n in DEEP_FIELDS if not (directory / "deep" / f"{n}.npy").is_file()]
    if missing:
        return [f"{directory.name}: missing {', '.join(missing)}"]
    try:
        arrays = load(directory)
    except ValueError as error:
        return [f"{directory.name}: unreadable: {error}"]
    found = _layout_errors(arrays, shape) or _sample_errors(arrays, shape[0] * shape[1])
    return [f"{directory.name}: {e}" for e in found]
