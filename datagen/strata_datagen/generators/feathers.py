"""Feathers grown on a skin: a shaft curve with barb curves either side, laid along a comb direction."""

from dataclasses import dataclass

import numpy as np

from strata_datagen.usdutil import FloatArray

VERSION = "feathers-0.1.0"


@dataclass(frozen=True)
class FeatherParams:
    count: int
    length_m: float
    length_jitter: float
    vane_width_m: float
    barbs: int
    barb_angle_deg: float
    lift: float
    curl: float
    shaft_width_m: float
    barb_width_m: float
    samples: int


@dataclass(frozen=True)
class Placement:
    """Feather roots on the skin with unit normals, unit comb tangents and texture coordinates."""

    roots: FloatArray
    normals: FloatArray
    combs: FloatArray
    st: FloatArray


@dataclass(frozen=True)
class Feathers:
    """Every shaft and barb as one curve of `samples` points; st is the feather root's."""

    curves: FloatArray
    widths: FloatArray
    st: FloatArray


def _unit(v: FloatArray) -> FloatArray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _shafts(place: Placement, params: FeatherParams, length: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Shaft points (feathers, samples, 3) and their tangents: they leave the skin and curl back to it."""
    u = np.linspace(0.0, 1.0, params.samples)[None, :, None]
    out = _unit(place.combs * (1.0 - params.lift) + place.normals * params.lift)[:, None]
    down = -place.normals[:, None]
    points = place.roots[:, None] + length[:, None, None] * (u * out + params.curl * u**2 * down)
    tangent = _unit(out + 2.0 * params.curl * u * down)
    return points, tangent


def build(place: Placement, params: FeatherParams, rng: np.random.Generator) -> Feathers:
    count = len(place.roots)
    length = params.length_m * (1.0 + params.length_jitter * rng.uniform(-1.0, 1.0, count))
    shaft, tangent = _shafts(place, params, length)
    side = _unit(np.cross(place.normals, place.combs))
    k = np.linspace(0.12, 0.98, params.barbs)
    at = np.rint(k * (params.samples - 1)).astype(np.int64)
    half = (params.vane_width_m * (length / params.length_m))[:, None] * np.sin(np.pi * k**0.75)[None, :]
    angle = np.radians(params.barb_angle_deg + rng.normal(0.0, 4.0, (count, params.barbs)))
    v = np.linspace(0.0, 1.0, params.samples)[None, None, :, None]
    barbs: list[FloatArray] = []
    for sign in (-1.0, 1.0):
        direction = _unit(
            np.cos(angle)[..., None] * tangent[:, at] + np.sin(angle)[..., None] * sign * side[:, None]
        )
        sag = 0.12 * -place.normals[:, None, None]
        barbs.append(
            shaft[:, at][:, :, None] + half[..., None, None] * (v * direction[:, :, None] + v**2 * sag)
        )
    curves = np.concatenate([shaft[:, None], *barbs], axis=1)
    taper = np.linspace(1.0, 0.3, params.samples)
    widths = np.empty(curves.shape[:3])
    widths[:, 0] = params.shaft_width_m * taper
    widths[:, 1:] = params.barb_width_m * taper
    per_feather = curves.shape[1]
    return Feathers(
        curves.reshape(-1, params.samples, 3),
        widths.reshape(-1, params.samples),
        np.repeat(place.st, per_feather, axis=0),
    )
