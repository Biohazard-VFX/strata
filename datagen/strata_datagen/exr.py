"""Read Karma flat (multipart) and deep EXRs into numpy arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenEXR
from numpy.typing import NDArray

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
    cells: list[NDArray[np.float32]] = [
        np.zeros(0, np.float32) if cell is None else np.asarray(cell, np.float32) for cell in pixels.ravel()
    ]
    counts = np.fromiter((c.size for c in cells), np.int64, len(cells))
    return counts, np.concatenate(cells) if cells else np.zeros(0, np.float32)


def read_deep(path: Path) -> Deep:
    part = OpenEXR.File(str(path), separate_channels=True).parts[0]
    counts: NDArray[np.int64] | None = None
    channels: dict[str, NDArray[np.float32]] = {}
    height, width = 0, 0
    for name, channel in part.channels.items():
        height, width = channel.pixels.shape[:2]
        counts, values = _flatten(channel.pixels)
        channels[name] = values
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
    return dict(OpenEXR.File(str(path)).parts[0].header)
