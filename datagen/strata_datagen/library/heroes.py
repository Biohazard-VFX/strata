"""External hero library: one directory per hero with a hero.json manifest and USD poses.

hero.json fields:
  key: library id; poses: {pose name: USD file relative to the hero directory};
  skin_prim: mesh prim path in each pose file; guide_prims: curve prim paths (may be empty);
  albedo: texture file relative to the hero directory; groom: GroomParams fields, optional when a
  coat (library/coat.py) replaces it;
  source, license, redistributable: provenance recorded in every shot manifest.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from strata_datagen import files
from strata_datagen.generators import groom
from strata_datagen.rng import stream
from strata_datagen.usdutil import FloatArray, IntArray, ints, new_stage, vec2f, vec3f


@dataclass(frozen=True)
class HeroManifest:
    key: str
    directory: Path
    poses: dict[str, str]
    skin_prim: str
    guide_prims: tuple[str, ...]
    albedo: str
    groom: groom.GroomParams | None
    source: str
    license: str
    redistributable: bool


def _groom(raw: dict[str, float]) -> groom.GroomParams:
    return groom.GroomParams(
        count=int(raw["count"]),
        segments=int(raw["segments"]),
        width_root_m=float(raw["width_root_m"]),
        width_tip_m=float(raw["width_tip_m"]),
        length_m=float(raw["length_m"]),
        length_jitter=float(raw["length_jitter"]),
        frizz_m=float(raw["frizz_m"]),
        neighbours=int(raw["neighbours"]),
        clump_count=int(raw.get("clump_count", 0)),
        clump_strength=float(raw.get("clump_strength", 0.0)),
    )


def load(directory: Path) -> HeroManifest:
    raw = cast("dict[str, object]", json.loads((directory / "hero.json").read_text()))
    return HeroManifest(
        key=str(raw["key"]),
        directory=directory,
        poses={str(k): str(v) for k, v in cast("dict[str, object]", raw["poses"]).items()},
        skin_prim=str(raw["skin_prim"]),
        guide_prims=tuple(str(p) for p in cast("list[object]", raw.get("guide_prims", []))),
        albedo=str(raw["albedo"]),
        groom=_groom(cast("dict[str, float]", raw["groom"])) if "groom" in raw else None,
        source=str(raw["source"]),
        license=str(raw["license"]),
        redistributable=bool(raw.get("redistributable", False)),
    )


def _world(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _transform(points: FloatArray, m: Gf.Matrix4d) -> FloatArray:
    rows = np.asarray(m, dtype=np.float64)
    return points @ rows[:3, :3] + rows[3, :3]


def _corner_st(mesh: UsdGeom.Mesh, corners: int) -> FloatArray:
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    values = np.asarray(primvar.ComputeFlattened(), dtype=np.float64).reshape(-1, 2)
    if primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying:
        return values
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    return values[indices] if values.shape[0] != corners else values


def fan(counts: IntArray) -> tuple[IntArray, IntArray]:
    """Triangle corner indices into the face-vertex list, fanning each polygon from its first corner."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tri = [(s, s + k, s + k + 1) for s, n in zip(starts, counts, strict=True) for k in range(1, n - 1)]
    corners = np.asarray(tri, dtype=np.int64)
    return corners, corners.ravel()


def read_skin(stage: Usd.Stage, path: str) -> tuple[groom.Skin, UsdGeom.Mesh]:
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    points = _transform(np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64), _world(mesh.GetPrim()))
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    corner_st = _corner_st(mesh, indices.size)
    corners, flat = fan(counts)
    triangles = indices[corners]
    return groom.Skin(points, triangles, corner_st[flat].reshape(-1, 2)), mesh


def read_guides(stage: Usd.Stage, paths: tuple[str, ...], samples: int) -> groom.Guides | None:
    curves: list[FloatArray] = []
    for path in paths:
        prim = UsdGeom.BasisCurves(stage.GetPrimAtPath(path))
        points = _transform(np.asarray(prim.GetPointsAttr().Get(), dtype=np.float64), _world(prim.GetPrim()))
        counts = np.asarray(prim.GetCurveVertexCountsAttr().Get(), dtype=np.int64)
        curves.extend(groom.resample(chunk, samples) for chunk in np.split(points, np.cumsum(counts)[:-1]))
    return groom.Guides(np.stack(curves)) if curves else None


def _write_skin(stage: Usd.Stage, skin: groom.Skin) -> None:
    mesh = UsdGeom.Mesh.Define(stage, "/hero/skin")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(vec3f(skin.points))
    mesh.CreateFaceVertexCountsAttr(ints(np.full(len(skin.triangles), 3)))
    mesh.CreateFaceVertexIndicesAttr(ints(skin.triangles.ravel()))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(vec2f(skin.corner_st))
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))


@dataclass(frozen=True)
class PreparedPose:
    file: Path
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    hairs: int


def prepare_pose(manifest: HeroManifest, pose: str, output: Path) -> PreparedPose:
    """Skin plus a generated groom for one pose, in a Y-up metre USD file with prims /hero/{skin,groom}."""
    if manifest.groom is None:
        raise ValueError(f"{manifest.key} has neither a groom nor a coat")
    source = Usd.Stage.Open(str(manifest.directory / manifest.poses[pose]))
    skin, _ = read_skin(source, manifest.skin_prim)
    guides = read_guides(source, manifest.guide_prims, manifest.groom.segments + 1)
    hair = groom.build(skin, guides, manifest.groom, stream(0, "groom", manifest.key, pose))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = new_stage(output)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/hero").GetPrim())
    _write_skin(stage, skin)
    groom.author(stage, "/hero/groom", hair)
    stage.GetRootLayer().Save()
    low, high = skin.points.min(axis=0), skin.points.max(axis=0)
    return PreparedPose(
        output,
        (float(low[0]), float(low[1]), float(low[2])),
        (float(high[0]), float(high[1]), float(high[2])),
        len(hair.curves),
    )


def copy_albedo(manifest: HeroManifest, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"albedo{Path(manifest.albedo).suffix}"
    with files.replacing(target) as temporary:
        shutil.copyfile(manifest.directory / manifest.albedo, temporary)
    return target
