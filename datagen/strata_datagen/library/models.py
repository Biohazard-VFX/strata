"""Canonical model assets: every mesh of a converted file merged into one Y-up, metre mesh at /asset.

The model is centred on its footprint and rests on y = 0, with face-varying st and normals.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom

from strata_datagen.usdutil import FloatArray, IntArray, ints, new_stage, vec2f, vec3f


@dataclass(frozen=True)
class MeshData:
    points: FloatArray
    counts: IntArray
    indices: IntArray
    st: FloatArray
    normals: FloatArray


@dataclass(frozen=True)
class ModelInfo:
    file: Path
    size_m: tuple[float, float, float]
    triangles: int


def _matrix(prim: Usd.Prim) -> FloatArray:
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(m, dtype=np.float64)


def _face_varying(
    mesh: UsdGeom.Mesh, values: FloatArray, interpolation: str, indices: IntArray
) -> FloatArray:
    if interpolation == UsdGeom.Tokens.faceVarying:
        return values
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        return values[indices]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    return (
        np.repeat(values, counts, axis=0)
        if interpolation == UsdGeom.Tokens.uniform
        else np.repeat(values[:1], indices.size, axis=0)
    )


def _read(mesh: UsdGeom.Mesh) -> MeshData:
    rows = _matrix(mesh.GetPrim())
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) @ rows[:3, :3] + rows[3, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    st_var = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    st = np.asarray(st_var.ComputeFlattened(), dtype=np.float64).reshape(-1, 2)
    st = _face_varying(mesh, st, st_var.GetInterpolation(), indices)
    raw_normals = mesh.GetNormalsAttr().Get()
    if raw_normals is None:
        normals = np.zeros((indices.size, 3))
    else:
        rotated = np.asarray(raw_normals, dtype=np.float64) @ np.linalg.inv(rows[:3, :3]).T
        normals = _face_varying(mesh, rotated, mesh.GetNormalsInterpolation(), indices)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return MeshData(points, counts, indices, st, normals)


def merge(stage: Usd.Stage) -> MeshData:
    parts = [_read(UsdGeom.Mesh(p)) for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    if not parts:
        raise ValueError(f"{stage.GetRootLayer().identifier}: no meshes")
    offsets = np.cumsum([0] + [p.points.shape[0] for p in parts[:-1]])
    return MeshData(
        points=np.concatenate([p.points for p in parts]),
        counts=np.concatenate([p.counts for p in parts]),
        indices=np.concatenate([p.indices + o for p, o in zip(parts, offsets, strict=True)]),
        st=np.concatenate([p.st for p in parts]),
        normals=np.concatenate([p.normals for p in parts]),
    )


def canonicalise(source: Path, target: Path) -> ModelInfo:
    data = merge(Usd.Stage.Open(str(source)))
    low, high = data.points.min(axis=0), data.points.max(axis=0)
    offset = np.array([-(low[0] + high[0]) / 2, -low[1], -(low[2] + high[2]) / 2])
    stage = new_stage(target)
    mesh = UsdGeom.Mesh.Define(stage, "/asset")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(vec3f(data.points + offset))
    mesh.CreateFaceVertexCountsAttr(ints(data.counts))
    mesh.CreateFaceVertexIndicesAttr(ints(data.indices))
    mesh.CreateNormalsAttr(vec3f(data.normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set(vec2f(data.st))
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    stage.GetRootLayer().Save()
    size = high - low
    return ModelInfo(target, (float(size[0]), float(size[1]), float(size[2])), int((data.counts - 2).sum()))
