from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.data.targets import reduce_samples
from strata_training.io import save_array
from strata_training.types import Array, Layers

exr: Any = import_module("OpenEXR")

FIELDS = ("offsets", "z", "alpha", "rgb", "ids")
SPLIT_EPSILON = 1e-6


@dataclass(frozen=True)
class DeepSamples:
    shape: tuple[int, int]
    offsets: Array
    z: Array
    alpha: Array
    rgb: Array
    ids: Array

    @property
    def counts(self) -> Array:
        return np.diff(self.offsets)

    @property
    def pixel(self) -> Array:
        return np.repeat(np.arange(self.shape[0] * self.shape[1]), self.counts)

    def select(self, keep: Array) -> "DeepSamples":
        return from_unsorted(
            self.shape,
            self.pixel[keep],
            self.z[keep],
            self.alpha[keep],
            self.rgb[keep],
            self.ids[keep],
        )

    def crop(self, top: int, left: int, height: int, width: int) -> "DeepSamples":
        rows = np.arange(top, top + height)[:, None] * self.shape[1]
        first: Array = self.offsets[rows + left].ravel()
        last: Array = self.offsets[rows + left + width].ravel()
        lengths: Array = last - first
        starts: Array = np.repeat(first - np.cumsum(lengths) + lengths, lengths)
        index: Array = starts + np.arange(int(lengths.sum()))
        within = self.offsets[rows + np.arange(left, left + width + 1)]
        counts = np.diff(within, axis=1).ravel()
        return DeepSamples(
            (height, width),
            np.concatenate(([0], np.cumsum(counts))),
            np.asarray(self.z[index]),
            np.asarray(self.alpha[index]),
            np.asarray(self.rgb[index]),
            np.asarray(self.ids[index]),
        )

    def scaled(self, factor: float) -> "DeepSamples":
        return DeepSamples(
            self.shape, self.offsets, self.z * factor, self.alpha, self.rgb, self.ids
        )

    def targets(self, layers: int = 4) -> Layers:
        return reduce_samples(
            self.z.astype(np.float64),
            self.alpha.astype(np.float64),
            self.rgb.astype(np.float64),
            self.counts,
            self.shape,
            layers,
        )


def from_unsorted(
    shape: tuple[int, int], pixel: Array, z: Array, alpha: Array, rgb: Array, ids: Array
) -> DeepSamples:
    order = np.lexsort((z, pixel))
    counts = np.bincount(pixel, minlength=shape[0] * shape[1])
    return DeepSamples(
        shape,
        np.concatenate(([0], np.cumsum(counts))),
        z[order],
        alpha[order],
        rgb[order],
        ids[order],
    )


def read_samples(path: Path) -> DeepSamples:
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        depth: Array = channels["Z"].pixels
        counts = np.fromiter((0 if v is None else len(v) for v in depth.flat), np.int64)
        values: dict[str, Array] = {}
        for name in ("Z", "A", "R", "G", "B", "instance_id"):
            chunks: list[Array] = [v for v in channels[name].pixels.flat if v is not None]
            values[name] = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    alpha = values["A"].astype(np.float32)
    ids = np.rint(
        np.divide(values["instance_id"], alpha, out=np.zeros_like(alpha), where=alpha > 0)
    ).astype(np.int64)
    rgb = np.stack([values[c] for c in "RGB"], axis=1).astype(np.float32)
    pixel = np.repeat(np.arange(counts.size), counts)
    return from_unsorted(depth.shape, pixel, values["Z"].astype(np.float32), alpha, rgb, ids)


def save_samples(directory: Path, samples: DeepSamples) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in FIELDS:
        save_array(directory / f"{name}.npy", getattr(samples, name))
    save_array(directory / "shape.npy", np.array(samples.shape, dtype=np.int64))


def load_samples(directory: Path) -> DeepSamples:
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in FIELDS}
    height, width = (int(v) for v in np.load(directory / "shape.npy"))
    return DeepSamples((height, width), **arrays)


def merge(parts: Sequence[DeepSamples]) -> tuple[DeepSamples, Array]:
    shape = parts[0].shape
    if any(p.shape != shape for p in parts):
        raise ValueError("Deep merge needs identical pixel grids")
    labels = np.concatenate([np.full(p.z.size, i) for i, p in enumerate(parts)])
    pixel = np.concatenate([p.pixel for p in parts])
    z = np.concatenate([p.z for p in parts])
    order = np.lexsort((labels, z, pixel))
    counts = np.bincount(pixel, minlength=shape[0] * shape[1])
    merged = DeepSamples(
        shape,
        np.concatenate(([0], np.cumsum(counts))),
        z[order],
        np.concatenate([p.alpha for p in parts])[order],
        np.concatenate([p.rgb for p in parts])[order],
        np.concatenate([p.ids for p in parts])[order],
    )
    return merged, labels[order]


def transmittance(samples: DeepSamples) -> Array:
    log_t = np.log1p(-np.minimum(np.clip(samples.alpha, 0, 1).astype(np.float64), 1 - 1e-15))
    cumulative = np.concatenate(([0.0], np.cumsum(log_t)))
    before = cumulative[:-1] - np.repeat(cumulative[samples.offsets[:-1]], samples.counts)
    return np.exp(before)


def contributions(samples: DeepSamples, labels: Array, groups: int) -> tuple[Array, Array]:
    height, width = samples.shape
    pixels = height * width
    index = labels * pixels + samples.pixel
    visible = transmittance(samples)
    coverage = np.bincount(index, visible * samples.alpha, groups * pixels)
    rgb = np.stack(
        [np.bincount(index, visible * samples.rgb[:, c], groups * pixels) for c in range(3)]
    )
    return rgb.reshape(3, groups, height, width).transpose(1, 0, 2, 3), coverage.reshape(
        groups, height, width
    )


def flatten(samples: DeepSamples) -> tuple[Array, Array]:
    rgb, coverage = contributions(samples, np.zeros(samples.z.size, dtype=np.int64), 1)
    return rgb[0], coverage[0]


@dataclass(frozen=True)
class Shot:
    deep: DeepSamples
    noisy: Array
    denoised: Array

    def crop(self, top: int, left: int, height: int, width: int) -> "Shot":
        window = (slice(None), slice(top, top + height), slice(left, left + width))
        return Shot(
            self.deep.crop(top, left, height, width), self.noisy[window], self.denoised[window]
        )


@dataclass(frozen=True)
class Layer:
    samples: DeepSamples
    shot: Shot | None
    matrix: Array = field(default_factory=lambda: np.eye(3))


@dataclass(frozen=True)
class Composition:
    samples: DeepSamples
    image: Array
    labels: Array


def compose(layers: Sequence[Layer], environment: Shot, environment_matrix: Array) -> Composition:
    merged, labels = merge([layer.samples for layer in layers])
    rgb, coverage = contributions(merged, labels, len(layers))
    image = np.zeros((3, *merged.shape))
    for g, layer in enumerate(layers):
        shot = layer.shot
        part = rgb[g] if shot is None else _explained(shot, rgb[g], coverage[g])
        image += np.tensordot(layer.matrix, part, axes=1)
    inserted = np.array([layer.shot is not environment for layer in layers])[labels]
    log_t = np.log1p(-np.minimum(merged.alpha[inserted].astype(np.float64), 1 - 1e-15))
    pixels = merged.shape[0] * merged.shape[1]
    through = np.exp(np.bincount(merged.pixel[inserted], log_t, pixels)).reshape(merged.shape)
    own_rgb, own_coverage = flatten(environment.deep)
    residual = _explained(
        environment, (environment.noisy - own_rgb) * through, (1 - own_coverage) * through
    )
    image += np.tensordot(environment_matrix, residual, axes=1)
    graded = np.stack([layer.matrix for layer in layers])[labels]
    colours = np.einsum("nij,nj->ni", graded, merged.rgb.astype(np.float64)).astype(np.float32)
    samples = DeepSamples(merged.shape, merged.offsets, merged.z, merged.alpha, colours, merged.ids)
    return Composition(samples, image.astype(np.float32), labels)


def _explained(shot: Shot, rgb: Array, coverage: Array) -> Array:
    return (
        shot.denoised
        * (rgb + SPLIT_EPSILON * coverage)
        / (np.maximum(shot.noisy, 0) + SPLIT_EPSILON)
    )
