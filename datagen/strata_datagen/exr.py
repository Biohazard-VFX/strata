"""Read Karma flat (multipart) and deep EXRs into numpy arrays."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import OpenEXR
from numpy.typing import NDArray

from strata_datagen import exrcore

Image = NDArray[np.float32]


@dataclass(frozen=True)
class Deep:
    """Per-pixel sample lists flattened in pixel order: pixel p owns values[offsets[p]:offsets[p + 1]]."""

    width: int
    height: int
    counts: NDArray[np.int64]
    channels: dict[str, NDArray[np.float32]]

    @property
    def offsets(self) -> NDArray[np.int64]:
        return np.concatenate([[0], np.cumsum(self.counts)])

    def pixel_index(self) -> NDArray[np.int64]:
        """For every sample, the flat index of the pixel that owns it."""
        return np.repeat(np.arange(self.counts.size), self.counts)


def _flatten(pixels: NDArray[np.generic]) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    cells = [cell for cell in pixels.ravel() if cell is not None]
    counts = np.fromiter((0 if c is None else c.size for c in pixels.ravel()), np.int64, pixels.size)
    return counts, np.concatenate(cells, dtype=np.float32) if cells else np.zeros(0, np.float32)


def read_deep(path: Path) -> Deep:
    metadata = header(path)
    if metadata["type"] == OpenEXR.deepscanline:
        low, high = cast("tuple[NDArray[np.int32], NDArray[np.int32]]", metadata["dataWindow"])
        width, height = int(high[0] - low[0] + 1), int(high[1] - low[1] + 1)
        counts, channels = exrcore.read(path, width, height, int(low[1]))
        return Deep(width, height, counts, channels)
    part = OpenEXR.File(str(path), separate_channels=True).parts[0]
    counts: NDArray[np.int64] | None = None
    channels: dict[str, NDArray[np.float32]] = {}
    height, width = 0, 0
    for name in list(part.channels):
        channel = part.channels.pop(name)
        height, width = channel.pixels.shape[:2]
        counts, values = _flatten(channel.pixels)
        channels[name] = values
        del channel
    if counts is None:
        raise ValueError(f"{path}: no deep channels")
    return Deep(width, height, counts, channels)


def read_flat(path: Path) -> dict[str, Image]:
    """All channels of every part, keyed `<part>.<channel>` except the beauty part `C`, keyed by channel."""
    out: dict[str, Image] = {}
    for part in OpenEXR.File(str(path), separate_channels=True).parts:
        for name, channel in part.channels.items():
            key = name if part.name() == "C" or "." in name else f"{part.name()}.{name}"
            out[key] = np.asarray(channel.pixels, np.float32)
    return out


def header(path: Path) -> dict[str, object]:
    return dict(OpenEXR.File(str(path), header_only=True).parts[0].header)
