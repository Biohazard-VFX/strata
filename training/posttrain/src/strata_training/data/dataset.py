import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strata_training.data import mapped
from strata_training.types import Array


@dataclass
class Frame:
    entry: dict[str, Any]
    feature_root: Path
    scale_mode: str = "raw"

    def __post_init__(self) -> None:
        if self.scale_mode not in {"raw", "oracle-aligned"}:
            raise ValueError("scale_mode must be raw or explicitly oracle-aligned")
        mapped.unpack(Path(self.entry["cache"]))
        mapped.unpack(self.root / "base.npz")
        context_path = self.root / "context.json"
        self.has_image = False
        if context_path.exists():
            context = json.loads(context_path.read_text())
            if "image_sha256" in context:
                image = np.load(self.root / "image.npy", mmap_mode="r")
                digest = hashlib.sha256(np.asarray(image, dtype=np.float32).tobytes()).hexdigest()
                if digest != context["image_sha256"]:
                    raise ValueError("Feature cache image identity mismatch")
                self.has_image = True

    @property
    def root(self) -> Path:
        return self.feature_root / self.entry["id"]

    @property
    def arrays(self) -> dict[str, Array]:
        values = mapped.arrays(Path(self.entry["cache"]))
        if self.has_image:
            return {
                **values,
                "target_beauty": values["beauty"],
                "beauty": mapped.image_array(self.root),
            }
        return values

    @property
    def features(self) -> Array:
        return mapped.feature_arrays(self.root)["features"]

    @property
    def pyramid(self) -> list[Array]:
        return mapped.feature_arrays(self.root)["pyramid"]

    @property
    def base(self) -> Array:
        prefix = "raw_" if self.scale_mode == "raw" else ""
        return mapped.arrays(self.root / "base.npz")[f"{prefix}inverse_depth"]

    @property
    def scale(self) -> float:
        prefix = "raw_" if self.scale_mode == "raw" else ""
        return float(mapped.arrays(self.root / "base.npz")[f"{prefix}metric_scale"])

    @property
    def mixed_y(self) -> Array:
        return self.arrays["mixed_y"]

    @property
    def mixed_x(self) -> Array:
        return self.arrays["mixed_x"]

    def crop(self, rng: np.random.Generator, size: int) -> tuple[int, int]:
        height, width = self.base.shape
        if rng.random() < 0.75 and self.mixed_y.size:
            i = int(rng.integers(self.mixed_y.size))
            cy, cx = int(self.mixed_y[i]), int(self.mixed_x[i])
        else:
            cy, cx = int(rng.integers(height)), int(rng.integers(width))
        return int(np.clip(cy - size // 2, 0, height - size)) // 32 * 32, int(
            np.clip(cx - size // 2, 0, width - size)
        ) // 32 * 32


def batch(
    frames: list[Frame],
    selections: list[tuple[int, int, int]],
    crop: int = 128,
    halo: int = 64,
    device: str = "cuda",
    decoder: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor]]:
    size = crop + 2 * halo

    def tensor(arrays: list[Array]) -> torch.Tensor:
        return torch.as_tensor(np.stack(arrays)).to(device).float()

    selected = [(frames[i], y, x) for i, y, x in selections]
    inputs = {
        "features": tensor([f.features[:, y : y + size, x : x + size] for f, y, x in selected]),
        "image": tensor(
            [f.arrays["beauty"][:, y : y + size, x : x + size] for f, y, x in selected]
        ),
        "base": tensor([f.base[None, y : y + size, x : x + size] for f, y, x in selected]),
        "scale": torch.tensor([f.scale for f, _, _ in selected], device=device)[
            :, None, None, None
        ],
    }
    targets = {
        key: tensor(
            [
                f.arrays.get("target_beauty", f.arrays[key])[
                    ..., y + halo : y + halo + crop, x + halo : x + halo + crop
                ]
                if key == "beauty"
                else f.arrays[key][..., y + halo : y + halo + crop, x + halo : x + halo + crop]
                for f, y, x in selected
            ]
        )
        for key in ("inverse_depth", "alpha", "rgb", "beauty", "mixed")
    }
    pyramid: list[torch.Tensor] = []
    if decoder:
        for level, stride in enumerate((4, 8, 16, 32)):
            pyramid.append(
                tensor(
                    [
                        f.pyramid[level][
                            :,
                            y // stride : (y + size) // stride,
                            x // stride : (x + size) // stride,
                        ]
                        for f, y, x in selected
                    ]
                )
            )
    return inputs, targets, pyramid


def normalization(frames: list[Frame]) -> tuple[Array, Array]:
    count = 0
    sums = np.zeros(frames[0].features.shape[0] + 4, dtype=np.float64)
    squares = np.zeros_like(sums)
    for frame in frames:
        values = (
            np.concatenate(
                (
                    frame.features[:, ::16, ::16],
                    frame.arrays["beauty"][:, ::16, ::16],
                    frame.base[None, ::16, ::16],
                )
            )
            .astype(np.float64)
            .reshape(sums.size, -1)
        )
        sums += values.sum(axis=1)
        squares += (values * values).sum(axis=1)
        count += values.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)
