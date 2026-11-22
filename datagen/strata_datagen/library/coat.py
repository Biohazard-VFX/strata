"""Layered creature coats: fur layers and feathers grown on skin regions, combed away from the head.

A creature hero's hero.json may carry a "coat" block next to the plain hero fields:
  front_vertex: skin vertex the coat is combed away from (the nose); poses share one topology;
  segments: points per curve minus one, shared by every layer so all curves merge into one prim;
  animated_density: hair and feather count factor for poses with per-frame skin points;
  fur: [{name, regions, zone, lift, droop, guides, groom: GroomParams fields}];
  feathers: [{name, regions, zone, params: FeatherParams fields except samples}];
  morph (optional): {length: [lo, hi], density: [lo, hi], period_s, key_step}, copied onto the
  prepared groom prim's customData so shots can vary the coat over time (generators/morph.py).
zone = {geodesic: [lo, hi] as fractions of the longest comb distance, min_up: lowest normal y}.
Pose skins carry float vertex primvars `region_<name>` (strata_datagen.regions); a layer grows on the
faces whose corners' mean weight in one of its regions is at least one half.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from strata_datagen import regions as skin_regions
from strata_datagen.generators import feathers, groom
from strata_datagen.library import heroes
from strata_datagen.rng import stream
from strata_datagen.usdutil import FloatArray, IntArray, ints, new_stage, vec2f, vec3f

Json = dict[str, object]


@dataclass(frozen=True)
class Zone:
    geodesic: tuple[float, float]
    min_up: float


@dataclass(frozen=True)
class FurLayer:
    name: str
    regions: tuple[str, ...]
    zone: Zone
    lift: float
    droop: float
    guides: int
    groom: groom.GroomParams


@dataclass(frozen=True)
class FeatherLayer:
    name: str
    regions: tuple[str, ...]
    zone: Zone
    params: feathers.FeatherParams


@dataclass(frozen=True)
class Coat:
    front_vertex: int
    segments: int
    animated_density: float
    fur: tuple[FurLayer, ...]
    feathers: tuple[FeatherLayer, ...]
    morph: Json | None


def _zone(raw: Json) -> Zone:
    lo, hi = cast("list[float]", raw["geodesic"])
    return Zone((float(lo), float(hi)), float(cast("float", raw["min_up"])))


def _fur(raw: Json, segments: int) -> FurLayer:
    params = {k: float(v) for k, v in cast("dict[str, float]", raw["groom"]).items()}
    return FurLayer(
        str(raw["name"]),
        tuple(cast("list[str]", raw["regions"])),
        _zone(cast("Json", raw["zone"])),
        float(cast("float", raw["lift"])),
        float(cast("float", raw["droop"])),
        int(cast("int", raw["guides"])),
        groom.GroomParams(
            count=int(params["count"]),
            segments=segments,
            width_root_m=params["width_root_m"],
            width_tip_m=params["width_tip_m"],
            length_m=params["length_m"],
            length_jitter=params["length_jitter"],
            frizz_m=params["frizz_m"],
            neighbours=int(params["neighbours"]),
            clump_count=int(params.get("clump_count", 0)),
            clump_strength=params.get("clump_strength", 0.0),
        ),
    )


def _feather(raw: Json, segments: int) -> FeatherLayer:
    params = {k: float(v) for k, v in cast("dict[str, float]", raw["params"]).items()}
    return FeatherLayer(
        str(raw["name"]),
        tuple(cast("list[str]", raw["regions"])),
        _zone(cast("Json", raw["zone"])),
        feathers.FeatherParams(
            count=int(params["count"]),
            length_m=params["length_m"],
            length_jitter=params["length_jitter"],
            vane_width_m=params["vane_width_m"],
            barbs=int(params["barbs"]),
            barb_angle_deg=params["barb_angle_deg"],
            lift=params["lift"],
            curl=params["curl"],
            shaft_width_m=params["shaft_width_m"],
            barb_width_m=params["barb_width_m"],
            samples=segments + 1,
        ),
    )


def load(directory: Path) -> Coat | None:
    raw = cast("Json", json.loads((directory / "hero.json").read_text())).get("coat")
    if raw is None:
        return None
    coat = cast("Json", raw)
    segments = int(cast("int", coat["segments"]))
    return Coat(
        int(cast("int", coat["front_vertex"])),
        segments,
        float(cast("float", coat["animated_density"])),
        tuple(_fur(f, segments) for f in cast("list[Json]", coat.get("fur", []))),
        tuple(_feather(f, segments) for f in cast("list[Json]", coat.get("feathers", []))),
        cast("Json | None", coat.get("morph")),
    )


def _unit(v: FloatArray) -> FloatArray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _welded(points: FloatArray) -> tuple[IntArray, int]:
    """One id per distinct position, so seams split for texture coordinates stay connected."""
    _, ids = np.unique(np.round(points / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    flat = np.asarray(ids, dtype=np.int64).ravel()
    return flat, int(flat.max()) + 1


def edge_graph(skin: groom.Skin) -> tuple[csr_matrix, IntArray]:
    """Welded skin edges as a sparse graph weighted by length, and each vertex's welded id."""
    weld, size = _welded(skin.points)
    tri = skin.triangles
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    length = np.linalg.norm(skin.points[a] - skin.points[b], axis=1) + 1e-9
    return coo_matrix((length, (weld[a], weld[b])), shape=(size, size)).tocsr(), weld


def geodesic(skin: groom.Skin, start: int) -> FloatArray:
    """Edge-path distance from vertex `start`; unreachable vertices are inf."""
    graph, weld = edge_graph(skin)
    return np.asarray(dijkstra(graph, directed=False, indices=int(weld[start])), dtype=np.float64)[weld]


def main_component(skin: groom.Skin) -> NDArray[np.bool_]:
    """Vertices of the largest connected piece of the skin (the body, not eyes or claws)."""
    graph, weld = edge_graph(skin)
    _, labels = connected_components(graph, directed=False)
    return (labels == np.argmax(np.bincount(labels)))[weld]


def comb_field(skin: groom.Skin, distance: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-vertex unit normals and unit tangents pointing down the distance gradient (away from the head)."""
    tri = skin.points[skin.triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    normal = cross / area2
    d = distance[skin.triangles]
    finite = np.all(np.isfinite(d), axis=1)
    d = np.where(np.isfinite(d), d, 0.0)
    gradient = np.zeros_like(normal)
    for k in range(3):
        edge = tri[:, (k + 2) % 3] - tri[:, (k + 1) % 3]
        gradient += d[:, k : k + 1] * np.cross(normal, edge) / area2
    gradient[~finite] = np.array([0.0, -1.0, 0.0])
    vertex_normal = np.zeros_like(skin.points)
    vertex_comb = np.zeros_like(skin.points)
    for k in range(3):
        np.add.at(vertex_normal, skin.triangles[:, k], cross)
        np.add.at(vertex_comb, skin.triangles[:, k], gradient * area2)
    vertex_normal = _unit(vertex_normal)
    tangent = vertex_comb - np.sum(vertex_comb * vertex_normal, axis=1, keepdims=True) * vertex_normal
    return vertex_normal, _unit(tangent)


@dataclass(frozen=True)
class Field:
    """The rest skin with its face regions, comb field and comb distance normalised to [0, 1]."""

    skin: groom.Skin
    regions: dict[str, FloatArray]
    normals: FloatArray
    combs: FloatArray
    distance: FloatArray


def field(skin: groom.Skin, regions: dict[str, FloatArray], front_vertex: int) -> Field:
    distance = geodesic(skin, front_vertex)
    normals, combs = comb_field(skin, distance)
    finite = np.isfinite(distance)
    scaled = np.where(finite, distance / max(float(distance[finite].max()), 1e-9), np.inf)
    return Field(skin, regions, normals, combs, scaled)


def select(f: Field, regions: tuple[str, ...], zone: Zone) -> tuple[groom.Skin, IntArray]:
    """The skin faces a layer grows on, and their indices into the full skin's triangles."""
    counts, corners = np.full(len(f.skin.triangles), 3), f.skin.triangles.ravel()
    wanted = np.zeros(len(f.skin.triangles), dtype=bool)
    for name in (r for r in regions if r in f.regions):
        wanted |= skin_regions.face_mask(f.regions[name], counts, corners)
    face_distance = f.distance[f.skin.triangles].mean(axis=1)
    lo, hi = zone.geodesic
    face_up = f.normals[f.skin.triangles].mean(axis=1)[:, 1]
    keep = wanted & (face_distance >= lo) & (face_distance <= hi) & (face_up >= zone.min_up)
    if not keep.any():
        raise ValueError(f"coat zone selects no skin in regions {regions}")
    corner_st = f.skin.corner_st.reshape(-1, 3, 2)[keep].reshape(-1, 2)
    return groom.Skin(f.skin.points, f.skin.triangles[keep], corner_st), np.flatnonzero(keep)


def _interpolate(values: FloatArray, part: groom.Skin, roots: groom.Roots) -> FloatArray:
    return _unit(np.einsum("nk,nkd->nd", roots.weights, values[part.triangles[roots.face]]))


def guides(f: Field, part: groom.Skin, layer: FurLayer, rng: np.random.Generator) -> groom.Guides:
    """Guide curves that leave the skin at `lift`, lie along the comb and droop under gravity."""
    roots = groom.sample_roots(part, layer.guides, rng)
    normal = _interpolate(f.normals, part, roots)
    comb = _interpolate(f.combs, part, roots)
    t = np.linspace(0.0, 1.0, layer.groom.segments + 1)[None, :, None]
    down = np.array([0.0, -1.0, 0.0])
    direction = _unit(
        normal[:, None] * layer.lift * (1.0 - t) + comb[:, None] * (1.0 - layer.lift) + down * layer.droop * t
    )
    steps = direction * (layer.groom.length_m / layer.groom.segments)
    path = np.concatenate([np.zeros_like(steps[:, :1]), np.cumsum(steps[:, :-1], axis=1)], axis=1)
    return groom.Guides(roots.position[:, None] + path)


def _scaled(params: groom.GroomParams, density: float) -> groom.GroomParams:
    count = max(1, round(params.count * density))
    clumps = min(params.clump_count, count)
    return groom.GroomParams(
        count,
        params.segments,
        params.width_root_m,
        params.width_tip_m,
        params.length_m,
        params.length_jitter,
        params.frizz_m,
        params.neighbours,
        clumps,
        params.clump_strength,
    )


def _feathers(f: Field, layer: FeatherLayer, density: float, rng: np.random.Generator) -> groom.Groom:
    part, kept = select(f, layer.regions, layer.zone)
    roots = groom.sample_roots(part, max(1, round(layer.params.count * density)), rng)
    placed = feathers.Placement(
        roots.position,
        _interpolate(f.normals, part, roots),
        _interpolate(f.combs, part, roots),
        roots.st,
    )
    built = feathers.build(placed, layer.params, rng)
    per = len(built.curves) // len(roots.position)
    face_index = np.repeat(kept[roots.face], per)
    return groom.Groom(
        built.curves, built.widths, built.st, face_index, np.repeat(roots.weights, per, axis=0)
    )


def grow(f: Field, coat: Coat, density: float, key: str) -> groom.Groom | None:
    """Every fur layer and feather layer of the coat as one groom bound to the full skin; None if bare."""
    parts: list[groom.Groom] = []
    for layer in coat.fur:
        part, kept = select(f, layer.regions, layer.zone)
        shapes = guides(f, part, layer, stream(0, "coat", key, layer.name, "guides"))
        grown = groom.build(part, shapes, _scaled(layer.groom, density), stream(0, "coat", key, layer.name))
        parts.append(groom.Groom(grown.curves, grown.widths, grown.st, kept[grown.face], grown.weights))
    parts.extend(_feathers(f, layer, density, stream(0, "coat", key, layer.name)) for layer in coat.feathers)
    if not parts:
        return None
    return groom.Groom(
        np.concatenate([p.curves for p in parts]),
        np.concatenate([p.widths for p in parts]),
        np.concatenate([p.st for p in parts]),
        np.concatenate([p.face for p in parts]),
        np.concatenate([p.weights for p in parts]),
    )


def _frames(points: FloatArray, triangles: IntArray) -> FloatArray:
    """Orthonormal row frames (edge, in-plane perpendicular, normal) per triangle."""
    tri = points[triangles]
    e1 = _unit(tri[:, 1] - tri[:, 0])
    n = _unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    return np.stack([e1, np.cross(n, e1), n], axis=1)


def deform(hair: groom.Groom, triangles: IntArray, rest: FloatArray, posed: FloatArray) -> FloatArray:
    """Hair points carried rigidly by their root triangle from the rest skin to a posed skin."""
    tri = triangles[hair.face]
    anchor_rest = np.einsum("nk,nkd->nd", hair.weights, rest[tri])
    anchor_posed = np.einsum("nk,nkd->nd", hair.weights, posed[tri])
    frame_rest = _frames(rest, tri)
    frame_posed = _frames(posed, tri)
    local = np.einsum("nsd,ned->nse", hair.curves - anchor_rest[:, None], frame_rest)
    return anchor_posed[:, None] + np.einsum("nse,ned->nsd", local, frame_posed)


def read_regions(stage: Usd.Stage, path: str) -> dict[str, FloatArray]:
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    return skin_regions.read(mesh, skin_regions.names(mesh))


def _skin_samples(stage: Usd.Stage, path: str) -> list[tuple[float, FloatArray]]:
    attribute = UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr()
    return [(time, np.asarray(attribute.Get(time), dtype=np.float64)) for time in attribute.GetTimeSamples()]


def _author_skin(
    stage: Usd.Stage,
    skin: groom.Skin,
    samples: list[tuple[float, FloatArray]],
    regions: dict[str, FloatArray],
) -> None:
    mesh = UsdGeom.Mesh.Define(stage, "/hero/skin")
    skin_regions.author(mesh, regions)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    points = mesh.CreatePointsAttr(vec3f(skin.points))
    for time, posed in samples:
        points.Set(vec3f(posed), Usd.TimeCode(time))
    mesh.CreateFaceVertexCountsAttr(ints(np.full(len(skin.triangles), 3)))
    mesh.CreateFaceVertexIndicesAttr(ints(skin.triangles.ravel()))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(vec2f(skin.corner_st))
    every = [skin.points, *(p for _, p in samples)]
    low = np.min([p.min(axis=0) for p in every], axis=0)
    high = np.max([p.max(axis=0) for p in every], axis=0)
    mesh.CreateExtentAttr(vec3f(np.stack([low, high])))


def _author_groom(stage: Usd.Stage, hair: groom.Groom, posed: list[tuple[float, FloatArray]]) -> None:
    curves = groom.author(stage, "/hero/groom", hair)
    if not posed:
        return
    points = curves.GetPointsAttr()
    for time, frame in posed:
        points.Set(vec3f(frame.reshape(-1, 3)), Usd.TimeCode(time))
    every = [hair.curves.reshape(-1, 3), *(f.reshape(-1, 3) for _, f in posed)]
    pad = float(np.max(hair.widths))
    low = np.min([p.min(axis=0) for p in every], axis=0) - pad
    high = np.max([p.max(axis=0) for p in every], axis=0) + pad
    curves.GetExtentAttr().Set(vec3f(np.stack([low, high])))


def prepare_pose(manifest: heroes.HeroManifest, pose: str, output: Path) -> heroes.PreparedPose:
    """Skin plus its layered coat for one pose; animated skins carry the coat along per frame."""
    coat = load(manifest.directory)
    if coat is None:
        raise ValueError(f"{manifest.key} has no coat")
    source = Usd.Stage.Open(str(manifest.directory / manifest.poses[pose]))
    skin, _ = heroes.read_skin(source, manifest.skin_prim)
    regions = read_regions(source, manifest.skin_prim)
    samples = _skin_samples(source, manifest.skin_prim)
    density = coat.animated_density if samples else 1.0
    hair = grow(field(skin, regions, coat.front_vertex), coat, density, f"{manifest.key}:{pose}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = new_stage(output)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/hero").GetPrim())
    _author_skin(stage, skin, samples, regions)
    if hair is not None:
        posed = [(time, deform(hair, skin.triangles, skin.points, frame)) for time, frame in samples]
        _author_groom(stage, hair, posed)
        if coat.morph is not None:
            stage.GetPrimAtPath("/hero/groom").SetCustomDataByKey("morph", coat.morph)
    stage.GetRootLayer().Save()
    every = [skin.points, *(p for _, p in samples)]
    low = np.min([p.min(axis=0) for p in every], axis=0)
    high = np.max([p.max(axis=0) for p in every], axis=0)
    return heroes.PreparedPose(
        output,
        (float(low[0]), float(low[1]), float(low[2])),
        (float(high[0]), float(high[1]), float(high[2])),
        0 if hair is None else len(hair.curves),
    )
