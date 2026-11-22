import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

from strata_training.augment.deep import DeepSamples, from_unsorted
from strata_training.augment.noise import gradient_noise
from strata_training.types import Array

KINDS = ("blobs", "grid", "stripes", "fence", "wires", "curves")
OCCLUDER_ID = -1
BLOB_STEP = 4
Shape = Callable[[tuple[int, int]], Array]


def box_overlap(distance: Array, width: float) -> Array:
    half = width / 2
    return np.clip(np.minimum(distance + 0.5, half) - np.maximum(distance - 0.5, -half), 0, 1)


def centres(size: tuple[int, int]) -> tuple[Array, Array]:
    rows, cols = np.mgrid[0 : size[0], 0 : size[1]].astype(np.float64) + 0.5
    return cols, rows


def line_family(angle: float, width: float, period: float, phase: float) -> Shape:
    normal = np.array([-np.sin(angle), np.cos(angle)])
    major = float(np.abs(normal).max())

    def shape(size: tuple[int, int]) -> Array:
        x, y = centres(size)
        offset = x * normal[0] + y * normal[1] - phase
        distance = np.mod(offset + period / 2, period) - period / 2
        return box_overlap(distance / major, width / major)

    return shape


def segment(start: Array, end: Array, width: float) -> Shape:
    length = float(np.hypot(*(end - start)))
    tangent = (end - start) / length
    normal = np.array([-tangent[1], tangent[0]])
    major = float(np.abs(normal).max())
    margin = width / major + 2

    def shape(size: tuple[int, int]) -> Array:
        coverage = np.zeros(size)
        low = np.floor(np.minimum(start, end) - margin).astype(int)
        high = np.ceil(np.maximum(start, end) + margin).astype(int)
        left, top = max(low[0], 0), max(low[1], 0)
        right, bottom = min(high[0], size[1]), min(high[1], size[0])
        if right <= left or bottom <= top:
            return coverage
        rows, cols = np.mgrid[top:bottom, left:right].astype(np.float64) + 0.5
        across = (cols - start[0]) * normal[0] + (rows - start[1]) * normal[1]
        along = (cols - start[0]) * tangent[0] + (rows - start[1]) * tangent[1]
        inside = np.clip(np.minimum(along + 0.5, length) - np.maximum(along - 0.5, 0), 0, 1)
        coverage[top:bottom, left:right] = box_overlap(across / major, width / major) * inside
        return coverage

    return shape


def union(shapes: Sequence[Shape]) -> Shape:
    def shape(size: tuple[int, int]) -> Array:
        clear = np.ones(size)
        for part in shapes:
            clear *= 1 - part(size)
        return 1 - clear

    return shape


def maximum(shapes: Sequence[Shape]) -> Shape:
    def shape(size: tuple[int, int]) -> Array:
        coverage = np.zeros(size)
        for part in shapes:
            np.maximum(coverage, part(size), out=coverage)
        return coverage

    return shape


def blobs(scale: float, threshold: float, seed: float) -> Shape:
    def shape(size: tuple[int, int]) -> Array:
        coarse = (size[0] // BLOB_STEP + 2, size[1] // BLOB_STEP + 2)
        rows, cols = np.mgrid[0 : coarse[0], 0 : coarse[1]].astype(np.float32) * BLOB_STEP
        depth = np.full(coarse, seed, dtype=np.float32)
        field = np.zeros(coarse, dtype=np.float32)
        for octave in range(3):
            frequency = np.float32(2.0**octave / scale)
            field += gradient_noise(cols * frequency, rows * frequency, depth) / 2.0**octave
        fine = zoom(field.astype(np.float64), BLOB_STEP, order=3)[: size[0], : size[1]]
        gy, gx = np.gradient(fine)
        slope = np.maximum(np.abs(gx) + np.abs(gy), 1e-9)
        return np.clip(0.5 + (fine - threshold) / slope, 0, 1)

    return shape


def motion_kernel(motion: tuple[float, float]) -> Array:
    reach = int(np.ceil(max(abs(motion[0]), abs(motion[1])) / 2)) + 1
    size = (2 * reach + 1, 2 * reach + 1)
    half = np.array(motion) / 2
    centre = np.array([reach + 0.5, reach + 0.5])
    kernel = segment(centre - half, centre + half + 1e-9, 1.0)(size)
    return kernel / kernel.sum()


def disc_kernel(radius: float) -> Array:
    reach = int(np.ceil(radius)) + 1
    y, x = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    disc = np.clip(radius + 0.5 - np.hypot(x, y), 0, 1)
    return disc / disc.sum()


def rasterize(
    shape: Shape, size: tuple[int, int], motion: tuple[float, float], defocus: float
) -> Array:
    coverage = shape(size)
    if np.hypot(*motion) >= 1:
        coverage = fftconvolve(coverage, motion_kernel(motion), mode="same")
    if defocus >= 0.5:
        coverage = fftconvolve(coverage, disc_kernel(defocus), mode="same")
    return np.clip(coverage, 0, 1)


@dataclass(frozen=True)
class Occluder:
    coverage: Array
    colour: Array
    depth: Array


def samples(occluders: Sequence[Occluder], size: tuple[int, int]) -> DeepSamples:
    pixels: list[Array] = []
    depths: list[Array] = []
    alphas: list[Array] = []
    colours: list[Array] = []
    for occluder in occluders:
        covered = np.flatnonzero(occluder.coverage > 1e-6)
        alpha = occluder.coverage.ravel()[covered]
        pixels.append(covered)
        depths.append(occluder.depth.ravel()[covered])
        alphas.append(alpha)
        colours.append(occluder.colour.reshape(3, -1)[:, covered].T * alpha[:, None])
    count = sum(p.size for p in pixels)
    return from_unsorted(
        size,
        np.concatenate(pixels).astype(np.int64),
        np.concatenate(depths).astype(np.float32),
        np.concatenate(alphas).astype(np.float32),
        np.concatenate(colours).astype(np.float32),
        np.full(count, OCCLUDER_ID, dtype=np.int64),
    )


def random_shape(kind: str, size: tuple[int, int], rng: np.random.Generator) -> Shape:
    height, width = size
    diagonal = float(np.hypot(height, width))
    angle = float(rng.uniform(0, np.pi))
    if kind == "blobs":
        return blobs(
            diagonal * rng.uniform(0.04, 0.2), rng.uniform(-0.1, 0.25), rng.uniform(0, 512)
        )
    if kind == "stripes":
        period = diagonal * rng.uniform(0.03, 0.15)
        return line_family(angle, period * rng.uniform(0.2, 0.6), period, rng.uniform(0, period))
    if kind in ("grid", "fence"):
        fence = kind == "fence"
        period = diagonal * (rng.uniform(0.01, 0.04) if fence else rng.uniform(0.02, 0.08))
        line = rng.uniform(0.3, 1.5) if fence else rng.uniform(0.5, 4.0)
        turn = rng.uniform(np.pi / 3, 2 * np.pi / 3) if fence else np.pi / 2
        return union(
            [line_family(angle + d, line, period, rng.uniform(0, period)) for d in (0.0, turn)]
        )
    if kind == "wires":
        wires: list[Shape] = []
        for _ in range(int(rng.integers(1, 6))):
            start = rng.uniform((0, 0), (width, height))
            direction = rng.uniform(0, 2 * np.pi)
            reach = diagonal * np.array([np.cos(direction), np.sin(direction)])
            wires.append(segment(start - reach, start + reach, rng.uniform(0.2, 2.5)))
        return union(wires)
    if kind == "curves":
        curves: list[Shape] = []
        for _ in range(int(rng.integers(1, 5))):
            points = [rng.uniform((0, 0), (width, height))]
            heading = rng.uniform(0, 2 * np.pi)
            for _ in range(int(rng.integers(8, 40))):
                heading += rng.normal(0, 0.35)
                step = diagonal * 0.03 * np.array([np.cos(heading), np.sin(heading)])
                points.append(points[-1] + step)
            thickness = rng.uniform(0.3, 3.0)
            pieces = [segment(a, b, thickness) for a, b in itertools.pairwise(points)]
            curves.append(maximum(pieces))
        return union(curves)
    raise ValueError(f"Unknown occluder kind {kind}")


def random_colour(
    size: tuple[int, int], luminance: float, textured: bool, rng: np.random.Generator
) -> Array:
    albedo = rng.uniform(0.03, 0.8, 3)
    light = luminance / 0.18 * rng.uniform(0.5, 1.5)
    colour = np.broadcast_to((albedo * light)[:, None, None], (3, *size)).copy()
    if textured:
        rows, cols = np.mgrid[0 : size[0], 0 : size[1]].astype(np.float32)
        frequency = np.float32(1 / rng.uniform(4, 40))
        depth = np.full(size, rng.uniform(0, 512), dtype=np.float32)
        texture = gradient_noise(cols * frequency, rows * frequency, depth)
        colour *= np.clip(1 + 1.2 * texture, 0.2, 2.0)
    return colour


def tilted_depth(size: tuple[int, int], depth: float, rng: np.random.Generator) -> Array:
    rows, cols = np.mgrid[0 : size[0], 0 : size[1]].astype(np.float64)
    tilt = rng.uniform(-0.3, 0.3, 2)
    return depth * np.exp(tilt[0] * (cols / size[1] - 0.5) + tilt[1] * (rows / size[0] - 0.5))
