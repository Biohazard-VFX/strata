"""Coats that change smoothly over a shot on fixed strands, and looping of cyclic pose clips.

Strand topology never changes: length and density vary by tapering curve widths to zero (a strand
beyond the current length fraction, or above the current density rank, has zero width). Parameters
are seeded smooth functions of time sampled at keys; USD interpolates widths linearly between keys.
"""

from dataclasses import dataclass

import numpy as np
from pxr import Sdf, Usd, UsdGeom

from strata_datagen.usdutil import FloatArray, floats

VERSION = "morph-0.1.0"
FPS = 24.0


@dataclass(frozen=True)
class Range:
    low: float
    high: float


@dataclass(frozen=True)
class CoatMorph:
    """Ranges the coat sweeps over a shot: visible length fraction and strand density fraction."""

    length: Range
    density: Range
    min_period_s: float
    key_step: int


def smooth(rng: np.random.Generator, times_s: FloatArray, span: Range, min_period_s: float) -> FloatArray:
    """A seeded smooth signal in [low, high]: three sinusoids with periods of at least min_period_s."""
    periods = min_period_s * rng.uniform(1.0, 3.0, 3)
    phases = rng.uniform(0.0, 2.0 * np.pi, 3)
    weights = rng.uniform(0.5, 1.0, 3)
    wave = np.sum(
        weights[:, None] * np.sin(2.0 * np.pi * times_s[None] / periods[:, None] + phases[:, None]), 0
    )
    unit = 0.5 + 0.5 * wave / weights.sum()
    return span.low + (span.high - span.low) * unit


def strand_widths(
    base: FloatArray, rank: FloatArray, length: float, density: float, softness: float = 0.08
) -> FloatArray:
    """Widths (strands, points) tapered to show `length` of every strand and the `density` lowest ranks."""
    t = np.linspace(0.0, 1.0, base.shape[1])[None]
    along = np.clip((length - t) / softness + 1.0, 0.0, 1.0)
    gate = np.clip((density - rank[:, None]) / softness + 1.0, 0.0, 1.0)
    return base * along * gate


def keys(frames: tuple[int, ...], step: int) -> tuple[int, ...]:
    """Key frames every `step` frames, always including the first and last frame."""
    inner = list(range(frames[0], frames[-1], step))
    return (*inner, frames[-1])


def author_coat(
    stage: Usd.Stage,
    groom_path: str,
    base: FloatArray,
    *,
    morph: CoatMorph,
    frames: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Time-sampled widths on a groom prim; returns the length and density per key for the manifest."""
    rank: FloatArray = rng.uniform(0.0, 1.0, base.shape[0])
    at = np.asarray(keys(frames, morph.key_step), dtype=np.float64)
    length = smooth(rng, at / FPS, morph.length, morph.min_period_s)
    density = smooth(rng, at / FPS, morph.density, morph.min_period_s)
    attribute = UsdGeom.BasisCurves(stage.OverridePrim(groom_path)).CreateWidthsAttr()
    for frame, value_l, value_d in zip(at, length, density, strict=True):
        attribute.Set(floats(strand_widths(base, rank, value_l, value_d)), Usd.TimeCode(float(frame)))
    return {"keys": at.tolist(), "length": length.round(5).tolist(), "density": density.round(5).tolist()}


def loop_times(period: int, first: int, last: int) -> list[tuple[float, float]]:
    """Value-clip times that replay clip frames 0..period over and over from stage frame `first`."""
    times: list[tuple[float, float]] = []
    start = first
    while start <= last:
        times += [(float(start), 0.0), (float(start + period), float(period))]
        start += period
    return times


def author_loop(prim: Usd.Prim, asset: str, clip_prim: str, *, period: int, first: int, last: int) -> None:
    """Drive a prim's time samples from a looping clip whose sample `period` equals sample 0."""
    clips = Usd.ClipsAPI(prim)
    clips.SetClipAssetPaths([Sdf.AssetPath(asset)])
    clips.SetClipPrimPath(clip_prim)
    clips.SetClipActive([(float(first), 0.0)])
    clips.SetClipTimes(loop_times(period, first, last))
