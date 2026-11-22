from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from strata_training.data.exr import decode_instance_ids
from strata_training.data.exr import read_samples as read_channels
from strata_training.data.targets import reduce_identified_samples
from strata_training.io import save_array
from strata_training.types import Array, Layers

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
        counts = np.bincount(self.pixel[keep], minlength=self.shape[0] * self.shape[1])
        return DeepSamples(
            self.shape,
            np.concatenate(([0], np.cumsum(counts))),
            np.asarray(self.z[keep]),
            np.asarray(self.alpha[keep]),
            np.asarray(self.rgb[keep]),
            np.asarray(self.ids[keep]),
        )

    def crop(self, top: int, left: int, height: int, width: int) -> "DeepSamples":
        rows = np.arange(top, top + height)[:, None] * self.shape[1]
        first: Array = self.offsets[rows + left].ravel()
        last: Array = self.offsets[rows + left + width].ravel()
        within = self.offsets[rows + np.arange(left, left + width + 1)]
        counts = np.diff(within, axis=1).ravel()

        def gather(values: Array) -> Array:
            return np.concatenate(
                [values[int(a) : int(b)] for a, b in zip(first, last, strict=True)]
            )

        return DeepSamples(
            (height, width),
            np.concatenate(([0], np.cumsum(counts))),
            gather(self.z),
            gather(self.alpha),
            gather(self.rgb),
            gather(self.ids),
        )

    def scaled(self, factor: float) -> "DeepSamples":
        return DeepSamples(
            self.shape, self.offsets, self.z * factor, self.alpha, self.rgb, self.ids
        )

    def targets(self, layers: int = 4) -> Layers:
        return reduce_identified_samples(
            self.z.astype(np.float64),
            self.alpha.astype(np.float64),
            self.rgb.astype(np.float64),
            self.counts,
            self.shape,
            self.ids,
            layers,
        )


def depth_order(pixel: Array, z: Array) -> Array:
    key = (pixel.astype(np.int64) << 32) | z.astype(np.float32).view(np.uint32).astype(np.int64)
    return np.argsort(key, kind="stable")


def from_unsorted(
    shape: tuple[int, int], pixel: Array, z: Array, alpha: Array, rgb: Array, ids: Array
) -> DeepSamples:
    if np.any(z <= 0):
        raise ValueError("Deep sample depths must be positive")
    order = depth_order(pixel, z)
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
    values, counts, shape = read_channels(
        path, ("Z", "A", "R", "G", "B", "instance_id", "coverage")
    )
    alpha = values["A"].astype(np.float32)
    ids = decode_instance_ids(values["instance_id"], values["coverage"]).astype(np.int64)
    rgb = np.stack([values[c] for c in "RGB"], axis=1).astype(np.float32)
    pixel = np.repeat(np.arange(counts.size), counts)
    return from_unsorted(shape, pixel, values["Z"].astype(np.float32), alpha, rgb, ids)


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
    order = depth_order(pixel, z)
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
    colours = np.array(merged.rgb, dtype=np.float32)
    for g, layer in enumerate(layers):
        if not np.array_equal(layer.matrix, np.eye(3)):
            mask = labels == g
            colours[mask] = colours[mask] @ layer.matrix.T.astype(np.float32)
    samples = DeepSamples(merged.shape, merged.offsets, merged.z, merged.alpha, colours, merged.ids)
    return Composition(samples, image.astype(np.float32), labels)


def _explained(shot: Shot, rgb: Array, coverage: Array) -> Array:
    return (
        shot.denoised
        * (rgb + SPLIT_EPSILON * coverage)
        / (np.maximum(shot.noisy, 0) + SPLIT_EPSILON)
    )
