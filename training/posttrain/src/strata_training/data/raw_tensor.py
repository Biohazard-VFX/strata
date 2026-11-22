"""Private file-backed raw tensors with descriptor-only interprocess transport."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from strata_training.augment.torch_deep import TensorSamples
from strata_training.torch_api import api


@dataclass(frozen=True, kw_only=True)
class MappedSamples(TensorSamples):
    """Own private COW mappings; pickle reopens immutable source rows, never sample bytes."""

    root: Path
    top: int
    left: int
    height: int
    width: int

    def __reduce__(self) -> tuple[Callable[..., "MappedSamples"], tuple[Path, int, int, int, int]]:
        return mapped_samples, (self.root, self.top, self.left, self.height, self.width)

    def transfer(self, function: Callable[[torch.Tensor], torch.Tensor]) -> TensorSamples:
        values = TensorSamples(
            self.shape,
            *(function(getattr(self, key)) for key in ("counts", "z", "alpha", "rgb", "ids")),
        )
        values.validate()
        if self.left == 0 and self.width == self.shape[1]:
            return values
        column = values.pixel % self.shape[1]
        keep = (column >= self.left) & (column < self.left + self.width)
        counts = values.counts.reshape(self.shape)[:, self.left : self.left + self.width].reshape(
            -1
        )
        return replace(
            values,
            shape=(self.height, self.width),
            counts=counts,
            z=values.z[keep],
            alpha=values.alpha[keep],
            rgb=values.rgb[keep],
            ids=values.ids[keep],
        )


def mapped_samples(root: Path, top: int, left: int, height: int, width: int) -> MappedSamples:
    """Own private source-row tensor mappings; caller must keep source files immutable."""
    shape = np.load(root / "shape.npy")
    if shape.shape != (2,) or shape.dtype != np.dtype(np.int64):
        raise ValueError("Raw samples require int64 native shape metadata")
    native_width = int(shape[1])
    offsets = np.load(root / "offsets.npy", mmap_mode="r")
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (int(shape.prod()) + 1,):
        raise ValueError("Raw samples require int64 native pixel offsets")
    begin, end = top * native_width, (top + height) * native_width
    first, last = int(offsets[begin]), int(offsets[end])
    if int(offsets[0]) != 0 or first < 0 or last < first:
        raise ValueError("Raw samples require nonnegative pixel offsets starting at zero")
    counts: torch.Tensor = api.from_numpy(np.diff(offsets[begin : end + 1]))
    arrays: list[torch.Tensor] = []
    for key in ("z", "alpha", "rgb", "ids"):
        values = np.load(root / f"{key}.npy", mmap_mode="c")
        dtype = np.dtype(np.int64 if key == "ids" else np.float32)
        expected = (int(offsets[-1]), 3) if key == "rgb" else (int(offsets[-1]),)
        if values.dtype != dtype or values.shape != expected or not values.flags.c_contiguous:
            raise ValueError(f"Raw samples require native C-order {dtype} {key} arrays")
        arrays.append(api.from_numpy(values[first:last]))
    return MappedSamples(
        (height, native_width),
        counts,
        *arrays,
        root=root,
        top=top,
        left=left,
        height=height,
        width=width,
    )
