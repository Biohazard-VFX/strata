from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from strata_contract.segmented import transmittance as sample_transmittance

from strata_training.augment.segmented import (
    contributions_arrays,
    inserted_through,
    merge_arrays,
    select_arrays,
)
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
        return DeepSamples(
            self.shape, *select_arrays(self.offsets, self.z, self.alpha, self.rgb, self.ids, keep)
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
            self.z,
            self.alpha,
            self.rgb,
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
    offsets, z, alpha, rgb, ids, labels = merge_arrays(
        tuple(np.asarray(p.offsets) for p in parts),
        tuple(np.asarray(p.z) for p in parts),
        tuple(np.asarray(p.alpha) for p in parts),
        tuple(np.asarray(p.rgb) for p in parts),
        tuple(np.asarray(p.ids) for p in parts),
    )
    return DeepSamples(shape, offsets, z, alpha, rgb, ids), labels


def transmittance(samples: DeepSamples) -> Array:
    return sample_transmittance(samples.alpha, samples.counts)


def contributions(samples: DeepSamples, labels: Array, groups: int) -> tuple[Array, Array]:
    height, width = samples.shape
    rgb, coverage = contributions_arrays(
        samples.offsets, samples.alpha, samples.rgb, labels, transmittance(samples), groups
    )
    return rgb.reshape(groups, 3, height, width), coverage.reshape(groups, height, width)


def flatten(samples: DeepSamples) -> tuple[Array, Array]:
    rgb, coverage = contributions(samples, np.zeros(samples.z.size, dtype=np.int64), 1)
    return rgb[0], coverage[0]


class Shot:
    """Borrow native images; materialize raw deep only on first use of a crop."""

    def __init__(self, deep: DeepSamples, noisy: Array, denoised: Array) -> None:
        self._deep: DeepSamples | None = deep
        self._source: tuple[Shot, int, int, int, int] | None = None
        self.noisy = noisy
        self.denoised = denoised

    @property
    def deep(self) -> DeepSamples:
        if self._deep is None:
            assert self._source is not None
            shot, top, left, height, width = self._source
            self._deep = shot.deep.crop(top, left, height, width)
        return self._deep

    def crop(self, top: int, left: int, height: int, width: int) -> "Shot":
        window = (slice(None), slice(top, top + height), slice(left, left + width))
        result = Shot.__new__(Shot)
        result._deep = None
        result._source = (self, top, left, height, width)
        result.noisy, result.denoised = self.noisy[window], self.denoised[window]
        return result


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
    inserted = np.array([layer.shot is not environment for layer in layers])
    through = inserted_through(merged.offsets, merged.alpha, labels, inserted).reshape(merged.shape)
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
