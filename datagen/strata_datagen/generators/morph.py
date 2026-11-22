"""Coats that change smoothly over a shot on fixed strands, and looping of cyclic pose clips.

Strand topology never changes: length and density vary by tapering curve widths to zero (a strand
beyond the current length fraction, or above the current density rank, has zero width). Parameters
are timeline curves sampled at keys; USD interpolates widths linearly between keys.
"""

from dataclasses import dataclass

import numpy as np
from pxr import Sdf, Usd, UsdGeom

from strata_datagen import timeline
from strata_datagen.usdutil import FloatArray, floats

VERSION = "morph-0.2.0"


@dataclass(frozen=True)
class Range:
    low: float
    high: float


@dataclass(frozen=True)
class CoatMorph:
    """Coat sweep ranges (visible length, strand density), timeline curve period, frames per key."""

    length: Range
    density: Range
    period_s: float
    key_step: int


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
    t = (at - at[0]) / timeline.FPS
    duration = float(t[-1]) + 1.0
    length = timeline.curve(rng, duration, morph.period_s, morph.length.low, morph.length.high)(t)
    density = timeline.curve(rng, duration, morph.period_s, morph.density.low, morph.density.high)(t)
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
