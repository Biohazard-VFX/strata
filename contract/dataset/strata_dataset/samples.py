"""Raw-sample sidecars: the memory-mappable deep samples and images of every frame.

A frame directory holds `noisy.npy` and `denoised.npy` (float32, 3 x H x W) and `deep/` with
`shape.npy` (int64 (2,)), `offsets.npy` (int64 (H*W+1,)), `z.npy`, `alpha.npy` (float32 (N,)),
`rgb.npy` (float32 (N, 3), premultiplied), `roles.npy` (float32 (N, 3), premultiplied foreground,
hero and fur coverage, `strata_dataset.roles`) and `ids.npy` (int64 (N,), best-effort instance ids,
-1 where the identity check failed): row-major pixels, samples front to back within a pixel.
Arrays are uncompressed little-endian C-order NPY, no pickles.
`identity.json` records the SHA-256 of every array file; its own SHA-256 is the frame's identity.
A published frame directory is immutable: it is never rewritten, only removed offline.
"""

import errno
import hashlib
import io
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from strata_dataset.identity import UNLABELLED

DEEP_FIELDS: dict[str, str] = {
    "shape": "<i8",
    "offsets": "<i8",
    "z": "<f4",
    "alpha": "<f4",
    "rgb": "<f4",
    "roles": "<f4",
    "ids": "<i8",
}
IMAGES = ("noisy", "denoised")
IDENTITY = "identity.json"
FILES: dict[str, str] = {f"{n}.npy": n for n in IMAGES} | {f"deep/{n}.npy": n for n in DEEP_FIELDS}


@dataclass(frozen=True)
class Frame:
    """A frame directory's arrays, keyed by file stem (deep fields unprefixed), and its identity."""

    identity: str
    arrays: dict[str, NDArray[np.generic]]


def frame_id(shot_id: str, frame: int) -> str:
    return f"{shot_id}_{frame:04d}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(arrays: Mapping[str, NDArray[np.generic]]) -> dict[str, bytes]:
    """The bytes of every file of a frame directory holding `arrays`, `identity.json` included."""
    if set(arrays) != set(FILES.values()):
        raise ValueError(f"sidecar arrays {sorted(arrays)}, expected {sorted(FILES.values())}")
    encoded: dict[str, bytes] = {}
    for name, stem in FILES.items():
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(arrays[stem]), allow_pickle=False)
        encoded[name] = buffer.getvalue()
    record = {"sha256": {name: _sha256(data) for name, data in encoded.items()}}
    encoded[IDENTITY] = (json.dumps(record, indent=1, sort_keys=True) + "\n").encode()
    return encoded


def identity(encoded: Mapping[str, bytes]) -> str:
    """The identity of a frame directory's files: the SHA-256 of its `identity.json`."""
    return _sha256(encoded[IDENTITY])


def recorded_identity(directory: Path) -> str | None:
    """The identity a frame directory records, without verifying its arrays; None when it has none."""
    path = directory / IDENTITY
    return _sha256(path.read_bytes()) if path.is_file() else None


def _open_directory(name: Path | str, parent: int | None = None) -> int:
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except OSError as error:
        if error.errno in (errno.ENOTDIR, errno.ELOOP):
            raise ValueError(f"{name} is a file or symlink, not a frame directory") from error
        raise


def _read(directory: int, name: str) -> bytes:
    with os.fdopen(os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory), "rb") as file:
        return file.read()


def _arrays(root: int, deep: int) -> Frame:
    found = {e.name for e in os.scandir(root)} | {f"deep/{e.name}" for e in os.scandir(deep)}
    if found != {*FILES, IDENTITY, "deep"}:
        raise ValueError(f"holds {sorted(found - {'deep'})}, expected {sorted({*FILES, IDENTITY})}")
    record = _read(root, IDENTITY)
    digests = cast("dict[str, str]", json.loads(record)["sha256"])
    if set(digests) != set(FILES):
        raise ValueError(f"{IDENTITY} lists {sorted(digests)}, expected {sorted(FILES)}")
    arrays: dict[str, NDArray[np.generic]] = {}
    for name, stem in FILES.items():
        parent, base = (deep, name.removeprefix("deep/")) if name.startswith("deep/") else (root, name)
        data = _read(parent, base)
        if _sha256(data) != digests[name]:
            raise ValueError(f"{name} does not match {IDENTITY}")
        arrays[stem] = np.load(io.BytesIO(data), allow_pickle=False)
    return Frame(_sha256(record), arrays)


def read(directory: Path) -> Frame:
    """Every array of a frame directory, read through one handle on it and verified against its identity."""
    root = _open_directory(directory)
    try:
        deep = _open_directory("deep", root)
        try:
            return _arrays(root, deep)
        finally:
            os.close(deep)
    finally:
        os.close(root)


def load(directory: Path) -> dict[str, NDArray[np.generic]]:
    """Every array of a frame directory, keyed by file stem (deep fields unprefixed); see `read`."""
    return read(directory).arrays


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
    if not errors and not all(np.isfinite(arrays[name]).all() for name in IMAGES):
        errors.append("images must be finite")
    return errors


def sample_errors(arrays: dict[str, NDArray[np.generic]], pixels: int, *, roundoff: float = 0.0) -> list[str]:
    """Validate borrowed deep values and offsets without mutation; calls are thread-safe."""
    offsets = arrays["offsets"].astype(np.int64)
    if offsets.shape != (pixels + 1,) or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        return ["offsets must start at 0, be nondecreasing and have H*W+1 entries"]
    count = int(offsets[-1])
    expected = {
        "z": (count,),
        "alpha": (count,),
        "rgb": (count, 3),
        "roles": (count, 3),
        "ids": (count,),
    }
    errors = [
        f"{name}: shape {arrays[name].shape}, expected {s}"
        for name, s in expected.items()
        if arrays[name].shape != s
    ]
    if errors:
        return errors
    z = cast("NDArray[np.float32]", arrays["z"])
    alpha = cast("NDArray[np.float32]", arrays["alpha"])
    if not (np.isfinite(z).all() and np.all(z > 0)):
        errors.append("z must be finite and positive")
    if not (np.isfinite(alpha).all() and np.all((alpha >= -roundoff) & (alpha <= 1 + roundoff))):
        errors.append("alpha must be in [0, 1]")
    if not np.isfinite(arrays["rgb"]).all():
        errors.append("rgb must be finite")
    roles = cast("NDArray[np.float32]", arrays["roles"])
    alpha_limit = alpha + roundoff if roundoff else alpha
    role_limits = roles[:, :-1] + roundoff if roundoff else roles[:, :-1]
    nested = np.all(roles[:, 0] <= alpha_limit) and np.all(roles[:, 1:] <= role_limits)
    if not (np.isfinite(roles).all() and np.all(roles >= -roundoff) and nested):
        errors.append("roles must satisfy 0 <= fur <= hero <= foreground <= alpha")
    if np.any(cast("NDArray[np.int64]", arrays["ids"]) < UNLABELLED):
        errors.append(f"ids must be nonnegative or {UNLABELLED} (unlabelled)")
    descending = z[1:] < z[:-1]
    boundaries = offsets[1:-1]
    boundaries = boundaries[(boundaries > 0) & (boundaries < count)]
    descending[boundaries - 1] = False
    if np.any(descending):
        errors.append("samples must be front to back within each pixel")
    return errors


def load_checked(directory: Path, shape: tuple[int, int]) -> tuple[dict[str, NDArray[np.generic]], list[str]]:
    """Owned verified arrays and contract violations, with each sidecar read once; calls are thread-safe."""
    missing = [n for n in (*FILES, IDENTITY) if not (directory / n).is_file()]
    if missing:
        return {}, [f"{directory.name}: missing {', '.join(missing)}"]
    try:
        arrays = load(directory)
    except ValueError as error:
        return {}, [f"{directory.name}: unreadable: {error}"]
    found = _layout_errors(arrays, shape) or sample_errors(arrays, shape[0] * shape[1])
    return arrays, [f"{directory.name}: {e}" for e in found]


def errors(directory: Path, shape: tuple[int, int]) -> list[str]:
    """Contract violations of one frame directory for an (H, W) frame; empty when it conforms."""
    return load_checked(directory, shape)[1]
