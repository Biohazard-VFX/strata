"""Hair and fur grooms on a skin mesh, interpolated from guide curves or grown from parameters."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom
from scipy.spatial import KDTree

from strata_datagen.usdutil import FloatArray, IntArray, floats, ints, vec2f, vec3f

VERSION = "groom-0.1.0"


@dataclass(frozen=True)
class Skin:
    """Triangulated skin in world space with per-corner texture coordinates."""

    points: FloatArray
    triangles: IntArray
    corner_st: FloatArray


@dataclass(frozen=True)
class Guides:
    """Guide curves resampled to a common point count, shape (curves, samples, 3)."""

    curves: FloatArray


@dataclass(frozen=True)
class GroomParams:
    count: int
    segments: int
    width_root_m: float
    width_tip_m: float
    length_m: float
    length_jitter: float
    frizz_m: float
    neighbours: int
    clump_count: int = 0
    clump_strength: float = 0.0


@dataclass(frozen=True)
class Groom:
    """Curves of equal point count; face and weights bind each root to its skin triangle."""

    curves: FloatArray
    widths: FloatArray
    st: FloatArray
    face: IntArray
    weights: FloatArray


def resample(points: FloatArray, samples: int) -> FloatArray:
    """A polyline resampled to `samples` points evenly spaced by arc length."""
    steps: FloatArray = np.linalg.norm(np.diff(points, axis=0), axis=1)
    lengths: FloatArray = np.concatenate([np.zeros(1), np.cumsum(steps)])
    total = float(lengths[-1])
    if total <= 0.0:
        return np.repeat(points[:1], samples, axis=0)
    targets: FloatArray = np.linspace(0.0, total, samples)
    return np.stack([np.interp(targets, lengths, points[:, i]) for i in range(3)], axis=-1)


@dataclass(frozen=True)
class Roots:
    """Skin points: position, unit face normal, texture coordinate, triangle and barycentric weights."""

    position: FloatArray
    normal: FloatArray
    st: FloatArray
    face: IntArray
    weights: FloatArray


def sample_roots(skin: Skin, count: int, rng: np.random.Generator) -> Roots:
    """Area-uniform points on the skin."""
    tri = skin.points[skin.triangles]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(normals, axis=1)
    face = rng.choice(len(area), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    weights = np.stack([1.0 - u - v, u, v], axis=-1)
    position = np.einsum("nk,nkd->nd", weights, tri[face])
    st = np.einsum("nk,nkd->nd", weights, skin.corner_st.reshape(-1, 3, 2)[face])
    normal = normals[face] / np.maximum(np.linalg.norm(normals[face], axis=1, keepdims=True), 1e-12)
    return Roots(position, normal, st, np.asarray(face, dtype=np.int64), weights)


def _from_guides(
    roots: FloatArray, guides: Guides, params: GroomParams
) -> tuple[FloatArray, NDArray[np.bool_]]:
    guide_roots = guides.curves[:, 0]
    shapes = guides.curves - guide_roots[:, None]
    tree = KDTree(guide_roots)
    spacing = float(np.median(tree.query(guide_roots, k=2)[0][:, 1]))
    distance, index = tree.query(roots, k=params.neighbours)
    distance, index = distance.reshape(len(roots), -1), index.reshape(len(roots), -1)
    weight = 1.0 / np.maximum(distance, 1e-6) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    blended = np.einsum("nk,nksd->nsd", weight, shapes[index])
    covered = distance[:, 0] < 4.0 * spacing
    return roots[:, None] + blended, covered


def _from_parameters(roots: FloatArray, normals: FloatArray, params: GroomParams) -> FloatArray:
    t = np.linspace(0.0, 1.0, params.segments + 1)[None, :, None]
    droop = np.array([0.0, -1.0, 0.0]) * 0.3 * t**2
    direction = normals[:, None] * (1.0 - 0.3 * t) + droop
    return roots[:, None] + direction * params.length_m * t


def _clump(curves: FloatArray, params: GroomParams, t: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Pull each hair towards its nearest clump-centre hair, more strongly towards the tip."""
    centres = rng.choice(len(curves), min(params.clump_count, len(curves)), replace=False)
    _, nearest = KDTree(curves[centres, 0]).query(curves[:, 0])
    leader = curves[centres[np.asarray(nearest, dtype=np.int64)]]
    return curves + (leader - curves) * (params.clump_strength * t[None, :, None])


def build(skin: Skin, guides: Guides | None, params: GroomParams, rng: np.random.Generator) -> Groom:
    roots = sample_roots(skin, params.count, rng)
    if guides is None:
        curves, keep = _from_parameters(roots.position, roots.normal, params), np.ones(params.count, bool)
    else:
        curves, keep = _from_guides(roots.position, guides, params)
    curves = curves[keep]
    scale = 1.0 + params.length_jitter * rng.uniform(-1.0, 1.0, (len(curves), 1, 1))
    curves = curves[:, :1] + (curves - curves[:, :1]) * scale
    t = np.linspace(0.0, 1.0, curves.shape[1])
    if params.clump_count > 0:
        curves = _clump(curves, params, t, rng)
    curves = curves + rng.normal(0.0, params.frizz_m, curves.shape) * t[None, :, None]
    widths = np.broadcast_to(
        params.width_root_m + (params.width_tip_m - params.width_root_m) * t, curves.shape[:2]
    )
    return Groom(curves, np.ascontiguousarray(widths), roots.st[keep], roots.face[keep], roots.weights[keep])


def author(stage: Usd.Stage, path: str, groom: Groom) -> UsdGeom.BasisCurves:
    count, samples, _ = groom.curves.shape
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr(ints(np.full(count, samples)))
    curves.CreatePointsAttr(vec3f(groom.curves.reshape(-1, 3)))
    curves.CreateWidthsAttr(floats(groom.widths.ravel()))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(curves).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.uniform
    )
    st.Set(vec2f(groom.st))
    curves.CreateExtentAttr(
        UsdGeom.Curves.ComputeExtent(curves.GetPointsAttr().Get(), curves.GetWidthsAttr().Get())
    )
    return curves
