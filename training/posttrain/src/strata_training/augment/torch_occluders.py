"""Native-grid procedural occluders evaluated entirely with device tensor math."""

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from strata_training.augment.camera import AP1_LUMINANCE
from strata_training.augment.occluders import KINDS, disc_kernel, motion_kernel
from strata_training.augment.pipeline import AugmentConfig
from strata_training.augment.torch_compose import TensorShot, geometry
from strata_training.augment.torch_deep import TensorSamples, merge
from strata_training.augment.torch_grain import gradient_noise
from strata_training.torch_api import api

Shape = Callable[[tuple[int, int]], torch.Tensor]


def centres(size: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(size[0], device=device, dtype=torch.float64) + 0.5
    x = torch.arange(size[1], device=device, dtype=torch.float64) + 0.5
    return x[None].expand(size), y[:, None].expand(size)


def box_overlap(distance: torch.Tensor, width: float) -> torch.Tensor:
    return ((distance + 0.5).clamp_max(width / 2) - (distance - 0.5).clamp_min(-width / 2)).clamp(
        0, 1
    )


def line_family(
    angle: float, width: float, period: float, phase: float, device: torch.device
) -> Shape:
    normal = (-float(np.sin(angle)), float(np.cos(angle)))
    major = max(abs(n) for n in normal)

    def shape(size: tuple[int, int]) -> torch.Tensor:
        x, y = centres(size, device)
        distance = (x * normal[0] + y * normal[1] - phase + period / 2) % period - period / 2
        return box_overlap(distance / major, width / major)

    return shape


def segment(
    start: np.ndarray[Any, Any], end: np.ndarray[Any, Any], width: float, device: torch.device
) -> Shape:
    length = float(np.hypot(*(end - start)))
    tangent = (end - start) / length
    normal = np.array([-tangent[1], tangent[0]])
    major = float(np.abs(normal).max())
    margin = width / major + 2
    low = np.floor(np.minimum(start, end) - margin).astype(int)
    high = np.ceil(np.maximum(start, end) + margin).astype(int)

    def shape(size: tuple[int, int]) -> torch.Tensor:
        x, y = centres(size, device)
        across = (x - start[0]) * normal[0] + (y - start[1]) * normal[1]
        along = (x - start[0]) * tangent[0] + (y - start[1]) * tangent[1]
        inside = ((along + 0.5).clamp_max(length) - (along - 0.5).clamp_min(0)).clamp(0, 1)
        bounds = (x >= max(low[0], 0)) & (x < min(high[0], size[1]))
        bounds &= (y >= max(low[1], 0)) & (y < min(high[1], size[0]))
        return box_overlap(across / major, width / major) * inside * bounds

    return shape


def combine(shapes: list[Shape], maximum: bool = False) -> Shape:
    def shape(size: tuple[int, int]) -> torch.Tensor:
        values = torch.stack([part(size) for part in shapes])
        return values.amax(0) if maximum else 1 - (1 - values).prod(0)

    return shape


def _spline_axis(length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.eye(length, device=device, dtype=torch.float64) * (4 / 6)
    index = torch.arange(length - 1, device=device)
    matrix[index, index + 1] = 1 / 6
    matrix[index + 1, index] = 1 / 6
    matrix[0, 1] = matrix[-1, -2] = 2 / 6
    coordinate = torch.arange(length * 4, device=device, dtype=torch.float64)
    coordinate *= (length - 1) / (length * 4 - 1)
    offset = coordinate[:, None] - torch.arange(length, device=device)[None]

    def basis(distance: torch.Tensor) -> torch.Tensor:
        x = distance.abs()
        return torch.where(
            x < 1, (4 - 6 * x**2 + 3 * x**3) / 6, torch.where(x < 2, (2 - x) ** 3 / 6, 0)
        )

    weight = basis(offset)
    weight[:, 1:] += basis(coordinate[:, None] + torch.arange(1, length, device=device)[None])
    weight[:, :-1] += basis(
        coordinate[:, None] - (2 * (length - 1) - torch.arange(length - 1, device=device))[None]
    )
    return matrix, weight


def blobs(scale: float, threshold: float, seed: float, device: torch.device) -> Shape:
    def shape(size: tuple[int, int]) -> torch.Tensor:
        coarse = (size[0] // 4 + 2, size[1] // 4 + 2)
        cols = torch.arange(coarse[1], device=device).float()[None].expand(coarse) * 4
        rows = torch.arange(coarse[0], device=device).float()[:, None].expand(coarse) * 4
        depth = torch.full_like(cols, seed)
        field = torch.zeros_like(cols)
        for octave in range(3):
            frequency = float(np.float32(2.0**octave / scale))
            field += gradient_noise(cols * frequency, rows * frequency, depth) / 2.0**octave
        ay, wy = _spline_axis(coarse[0], device)
        ax, wx = _spline_axis(coarse[1], device)
        coefficients: torch.Tensor = api.linalg.solve(ay, field.double())
        coefficients = api.linalg.solve(ax, coefficients.T).T
        fine = (wy @ coefficients @ wx.T)[: size[0], : size[1]]
        gy, gx = torch.gradient(fine)
        slope = (gx.abs() + gy.abs()).clamp_min(1e-9)
        return (0.5 + (fine - threshold) / slope).clamp(0, 1)

    return shape


def random_shape(
    kind: str, size: tuple[int, int], rng: np.random.Generator, device: torch.device
) -> Shape:
    height, width = size
    diagonal = float(np.hypot(height, width))
    angle = float(rng.uniform(0, np.pi))
    if kind == "blobs":
        return blobs(
            diagonal * rng.uniform(0.04, 0.2), rng.uniform(-0.1, 0.25), rng.uniform(0, 512), device
        )
    if kind == "stripes":
        period = diagonal * rng.uniform(0.03, 0.15)
        return line_family(
            angle, period * rng.uniform(0.2, 0.6), period, rng.uniform(0, period), device
        )
    if kind in ("grid", "fence"):
        fence = kind == "fence"
        period = diagonal * (rng.uniform(0.01, 0.04) if fence else rng.uniform(0.02, 0.08))
        line = rng.uniform(0.3, 1.5) if fence else rng.uniform(0.5, 4)
        turn = rng.uniform(np.pi / 3, 2 * np.pi / 3) if fence else np.pi / 2
        return combine(
            [
                line_family(angle + d, line, period, rng.uniform(0, period), device)
                for d in (0.0, turn)
            ]
        )
    if kind == "wires":
        wires: list[Shape] = []
        for _ in range(int(rng.integers(1, 6))):
            start = rng.uniform((0, 0), (width, height))
            direction = rng.uniform(0, 2 * np.pi)
            reach = diagonal * np.array([np.cos(direction), np.sin(direction)])
            wires.append(segment(start - reach, start + reach, rng.uniform(0.2, 2.5), device))
        return combine(wires)
    if kind == "curves":
        curves: list[Shape] = []
        for _ in range(int(rng.integers(1, 5))):
            points = [rng.uniform((0, 0), (width, height))]
            heading = rng.uniform(0, 2 * np.pi)
            for _ in range(int(rng.integers(8, 40))):
                heading += rng.normal(0, 0.35)
                points.append(
                    points[-1] + diagonal * 0.03 * np.array([np.cos(heading), np.sin(heading)])
                )
            thickness = rng.uniform(0.3, 3)
            curves.append(
                combine(
                    [segment(a, b, thickness, device) for a, b in itertools.pairwise(points)],
                    True,
                )
            )
        return combine(curves)
    raise ValueError(f"Unknown occluder kind {kind}")


def rasterize(
    shape: Shape, size: tuple[int, int], motion: tuple[float, float], defocus: float
) -> torch.Tensor:
    coverage = shape(size)
    for enabled, kernel in (
        (np.hypot(*motion) >= 1, lambda: motion_kernel(motion)),
        (defocus >= 0.5, lambda: disc_kernel(defocus)),
    ):
        if enabled:
            weights = torch.tensor(kernel(), device=coverage.device, dtype=coverage.dtype)
            extent = tuple(a + b - 1 for a, b in zip(coverage.shape, weights.shape, strict=True))
            spectrum = api.fft.rfft2(coverage, s=extent) * api.fft.rfft2(weights, s=extent)
            filtered: torch.Tensor = api.fft.irfft2(spectrum, s=extent)
            top, left = weights.shape[0] // 2, weights.shape[1] // 2
            coverage = filtered[top : top + size[0], left : left + size[1]]
    return coverage.clamp(0, 1)


def make_occluders(
    shot: TensorShot, rng: np.random.Generator, config: AugmentConfig, record: dict[str, Any]
) -> TensorSamples:
    assert shot.deep is not None
    size, device = shot.deep.shape, shot.denoised.device
    nearest, farthest, filled = geometry(shot.deep)
    near = float(torch.quantile(nearest[filled], 0.01))
    far = float(torch.quantile(farthest[filled], 0.95))
    luminance = float(
        torch.einsum(
            "c,chw->hw", torch.tensor(AP1_LUMINANCE, device=device), shot.denoised.double()
        ).mean()
    )
    parts: list[TensorSamples] = []
    kinds: list[str] = []
    x, y = centres(size, device)
    for _ in range(int(rng.integers(config.occluder_count[0], config.occluder_count[1] + 1))):
        kind = str(rng.choice(KINDS))
        motion = (0.0, 0.0)
        if rng.random() < config.occluder_motion:
            motion = (float(rng.normal(0, 8)), float(rng.normal(0, 8)))
        defocus = float(rng.uniform(0.5, 6)) if rng.random() < config.occluder_defocus else 0.0
        coverage = rasterize(random_shape(kind, size, rng, device), size, motion, defocus)
        textured = bool(rng.random() < config.occluder_textured)
        albedo = rng.uniform(0.03, 0.8, 3)
        light = luminance / 0.18 * rng.uniform(0.5, 1.5)
        colour = torch.tensor(albedo * light, device=device)[:, None, None].expand(3, *size)
        if textured:
            frequency = float(np.float32(1 / rng.uniform(4, 40)))
            z = torch.full(size, float(rng.uniform(0, 512)), device=device)
            texture = gradient_noise(
                (x.float() - 0.5) * frequency, (y.float() - 0.5) * frequency, z
            )
            colour = colour * (1 + 1.2 * texture).clamp(0.2, 2)
        if rng.random() < config.occluder_front:
            depth = near * float(rng.uniform(0.3, 0.9))
        else:
            depth = float(np.exp(rng.uniform(np.log(near), np.log(max(far, near * 1.01)))))
        tilt = rng.uniform(-0.3, 0.3, 2)
        z = (
            depth
            * (tilt[0] * ((x - 0.5) / size[1] - 0.5) + tilt[1] * ((y - 0.5) / size[0] - 0.5)).exp()
        )
        keep = coverage.ravel() > 1e-6
        alpha = coverage.ravel()[keep].float()
        parts.append(
            TensorSamples(
                size,
                keep.long(),
                z.ravel()[keep].float(),
                alpha,
                colour.reshape(3, -1).T[keep].float() * alpha[:, None],
                torch.full_like(alpha, -1, dtype=torch.long),
                torch.stack((alpha, torch.zeros_like(alpha), torch.zeros_like(alpha)), dim=1),
            )
        )
        kinds.append(kind)
    record["occluders"] = kinds
    return merge(parts)[0]
