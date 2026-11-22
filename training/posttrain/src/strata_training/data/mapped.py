"""Atomic, one-time NPZ migration and bounded, read-only array mappings."""

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from strata_training.io import save_array
from strata_training.types import Array


def unpack(path: Path) -> Path:
    """Publish an immutable NPY directory; callers must not replace source caches in place."""
    destination = path.with_suffix(path.suffix + ".mmap")
    stat = path.stat()
    identity = (stat.st_size, stat.st_mtime_ns)
    if destination.exists():
        if tuple(np.load(destination / "source.npy")) != identity:
            raise ValueError(f"Source cache changed; remove stale mapping directory {destination}")
    else:
        with TemporaryDirectory(dir=path.parent, prefix=f".{path.stem}.") as temporary:
            root = Path(temporary)
            with np.load(path) as archive:
                for name in archive.files:
                    save_array(root / f"{name}.npy", archive[name])
                if "mixed" in archive.files:
                    y, x = np.nonzero(archive["mixed"])
                    save_array(root / "mixed_y.npy", y)
                    save_array(root / "mixed_x.npy", x)
            save_array(root / "source.npy", np.array(identity, dtype=np.int64))
            try:
                root.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    return destination


@lru_cache(maxsize=2)
def arrays(path: Path) -> dict[str, Array]:
    """Borrow immutable mappings, bounded to two archives per process; never mutate values."""
    root = unpack(path)
    return {p.stem: np.load(p, mmap_mode="r") for p in root.glob("*.npy") if p.stem != "source"}


@lru_cache(maxsize=2)
def image_array(root: Path) -> Array:
    return np.load(root / "image.npy", mmap_mode="r")


@lru_cache(maxsize=2)
def feature_arrays(root: Path) -> dict[str, Any]:
    """Borrow at most two frames of file mappings per process."""
    return {
        "features": np.load(root / "features.npy", mmap_mode="r"),
        "pyramid": [np.load(root / f"pyramid_{i}.npy", mmap_mode="r") for i in range(4)],
        "base": arrays(root / "base.npz"),
    }
