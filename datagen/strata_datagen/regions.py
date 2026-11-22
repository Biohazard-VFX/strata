"""Named skin regions as float vertex primvars `region_<name>` (0..1), the one region form for every hero.

Weights are soft: a vertex on a region's border carries the share of its faces inside the region, so
the same values serve as groom density and as masks. Faces belong to a region when the mean weight
of their corners is at least one half.
"""

import re

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom

from strata_datagen.usdutil import FloatArray, IntArray, floats

PREFIX = "region_"
NAME = re.compile(r"[a-z][a-z0-9_]*")
Mask = NDArray[np.bool_]


def from_faces(faces: Mask, counts: IntArray, indices: IntArray, points: int) -> FloatArray:
    """Vertex weights from a face mask: the share of each vertex's faces that are in the region."""
    face = np.repeat(np.arange(len(counts)), counts)
    inside = np.bincount(indices, weights=faces[face].astype(np.float64), minlength=points)
    total = np.bincount(indices, minlength=points).astype(np.float64)
    return np.divide(inside, total, out=np.zeros(points), where=total > 0)


def face_mask(weight: FloatArray, counts: IntArray, indices: IntArray) -> Mask:
    """Faces whose corners' mean weight is at least one half."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.add.reduceat(weight[indices], starts) >= 0.5 * counts


def author(mesh: UsdGeom.Mesh, weights: dict[str, FloatArray]) -> None:
    primvars = UsdGeom.PrimvarsAPI(mesh)
    for name, weight in sorted(weights.items()):
        if not NAME.fullmatch(name):
            raise ValueError(f"region name {name!r} is not a lower-case identifier")
        primvar = primvars.CreatePrimvar(
            f"{PREFIX}{name}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(floats(weight))


def names(mesh: UsdGeom.Mesh) -> tuple[str, ...]:
    found = (p.GetPrimvarName() for p in UsdGeom.PrimvarsAPI(mesh).GetPrimvars())
    return tuple(sorted(str(n).removeprefix(PREFIX) for n in found if str(n).startswith(PREFIX)))


def read(mesh: UsdGeom.Mesh, wanted: tuple[str, ...]) -> dict[str, FloatArray]:
    """The named regions of a skin, checked: float, vertex interpolation, one weight per point, 0..1."""
    primvars = UsdGeom.PrimvarsAPI(mesh)
    points = len(mesh.GetPointsAttr().Get(Usd.TimeCode.EarliestTime()))
    out: dict[str, FloatArray] = {}
    for name in wanted:
        primvar = primvars.GetPrimvar(f"{PREFIX}{name}")
        path = mesh.GetPath()
        if not primvar:
            raise ValueError(f"{path} has no primvar {PREFIX}{name}")
        if primvar.GetTypeName() != Sdf.ValueTypeNames.FloatArray:
            raise ValueError(f"{path} {PREFIX}{name} is {primvar.GetTypeName()}, not float[]")
        if primvar.GetInterpolation() != UsdGeom.Tokens.vertex:
            raise ValueError(
                f"{path} {PREFIX}{name} has {primvar.GetInterpolation()} interpolation, not vertex"
            )
        weight = np.asarray(primvar.Get(), dtype=np.float64)
        if weight.shape != (points,):
            raise ValueError(f"{path} {PREFIX}{name} has {weight.size} values for {points} points")
        if weight.min() < 0.0 or weight.max() > 1.0:
            raise ValueError(f"{path} {PREFIX}{name} leaves [0, 1]")
        out[name] = weight
    return out
