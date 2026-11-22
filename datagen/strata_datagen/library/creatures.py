"""Creature hero library prep: generate or import bodies, orient them, and write hero folders with coats.

Recipes (a JSON file outside the repo) list creatures. Each one is either an Infinigen creature
(source "infinigen", run by blender/infinigen_creature.py in Infinigen's Python) or an authored
glTF animal (source "gltf", run by blender/gltf_creature.py in Blender). Both write a raw export
(skin.npz, albedo.png, creature.json); this module turns it into poses/<pose>.usdc, textures/albedo.png
and hero.json with a coat sampled from coat classes.

Recipe fields: key, source, job (passed to the Blender script), poses {name: {start, frames, loop,
shortest}} (a loop pose keeps one detected cycle, closed so its last frame equals its first),
length_m (body length along its heading), coat (coat class names), forward (heading in Blender
axes when the body has no head region), provenance {source, license, redistributable}.
"""

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from pxr import Sdf, UsdGeom, Vt

from strata_datagen import files
from strata_datagen.generators import groom
from strata_datagen.library import coat, heroes
from strata_datagen.rng import stream
from strata_datagen.usdutil import FloatArray, IntArray, ints, new_stage, vec2f, vec3f

BLENDER_DIR = Path(__file__).resolve().parents[2] / "blender"
SKIN_PRIM = "/creature/skin"
HEAD_REGIONS = ("head", "jaw", "head_detail", "beak")
FURRED = ("head", "jaw", "head_detail", "body", "tail", "leg", "lid")
SEGMENTS = 7
MAX_HAIRS = 400_000
MAX_THICKEN = 2.5
Json = dict[str, object]


@dataclass(frozen=True)
class Raw:
    """A Blender export in Y-up metres: points (frames, vertices, 3), triangles and their data."""

    points: FloatArray
    triangles: IntArray
    corner_st: FloatArray
    regions: IntArray
    names: tuple[str, ...]
    frames: tuple[int, ...]
    info: Json


def read_raw(folder: Path) -> Raw:
    data = np.load(folder / "skin.npz")
    info = cast("Json", json.loads((folder / "creature.json").read_text()))
    blender = np.asarray(data["points"], dtype=np.float64)
    points = np.stack([blender[..., 0], blender[..., 2], -blender[..., 1]], axis=-1)
    counts = np.asarray(data["counts"], dtype=np.int64)
    corners, _ = heroes.fan(counts)
    triangles = np.asarray(data["indices"], dtype=np.int64)[corners]
    face = np.repeat(np.arange(len(counts)), np.maximum(counts - 2, 0))
    uv = np.asarray(data["uv"], dtype=np.float64)
    return Raw(
        points,
        triangles,
        uv[corners.ravel()],
        np.asarray(data["regions"], dtype=np.int64)[face],
        tuple(cast("list[str]", info["regions"])),
        tuple(int(f) for f in cast("list[int]", info["frames"])),
        info,
    )


def _heading(raw: Raw, forward: tuple[float, float, float] | None) -> FloatArray:
    if forward is not None:
        x, y, z = forward
        return np.array([x, z, -y], dtype=np.float64)
    head = np.isin(raw.regions, [raw.names.index(r) for r in HEAD_REGIONS if r in raw.names])
    if not head.any():
        raise ValueError("a body without a head region needs a forward axis")
    rest = raw.points[0]
    return rest[np.unique(raw.triangles[head])].mean(axis=0) - rest.mean(axis=0)


def orient(raw: Raw, forward: tuple[float, float, float] | None, length_m: float) -> FloatArray:
    """Rotate about +Y so the heading is +Z, scale to `length_m` along it, centre the rest footprint."""
    heading = _heading(raw, forward)
    yaw = np.arctan2(heading[0], heading[2])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    points = raw.points @ rotation
    rest = points[0]
    scale = length_m / float(rest[:, 2].max() - rest[:, 2].min())
    low, high = rest.min(axis=0), rest.max(axis=0)
    centre = np.array([(low[0] + high[0]) / 2.0, low[1], (low[2] + high[2]) / 2.0])
    return (points - centre) * scale


def in_place(clip: FloatArray) -> FloatArray:
    """Remove the clip's horizontal drift so the hero generator alone moves the creature."""
    drift = clip.mean(axis=1) - clip[0].mean(axis=0)
    drift[:, 1] = 0.0
    return clip - drift[:, None]


def loop_period(clip: FloatArray, shortest: int) -> int:
    """Frames per cycle: the offset (at least `shortest`) whose next three poses match the first three."""
    offsets = range(shortest, len(clip) - 2)
    error = [float(np.abs(clip[k : k + 3] - clip[:3]).mean()) for k in offsets]
    return shortest + int(np.argmin(error))


def close_loop(clip: FloatArray, period: int) -> FloatArray:
    """One cycle of `period` + 1 frames, its end-to-start residual spread linearly: last equals first."""
    cycle = clip[: period + 1].copy()
    residual = cycle[0] - cycle[period]
    weight = np.linspace(0.0, 1.0, period + 1)[:, None, None]
    return cycle + weight * residual


def write_pose(path: Path, raw: Raw, clip: FloatArray) -> None:
    """A pose file: the skin mesh at the clip's first frame, time-sampled from 0 when it moves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with files.replacing(path) as temporary:
        stage = new_stage(temporary)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/creature").GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, SKIN_PRIM)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        points = mesh.CreatePointsAttr(vec3f(clip[0]))
        if len(clip) > 1:
            for time, frame in enumerate(clip):
                points.Set(vec3f(frame), float(time))
        mesh.CreateFaceVertexCountsAttr(ints(np.full(len(raw.triangles), 3)))
        mesh.CreateFaceVertexIndicesAttr(ints(raw.triangles.ravel()))
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set(vec2f(raw.corner_st))
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "region", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform
        ).Set(ints(raw.regions))
        mesh.GetPrim().SetCustomDataByKey("regions", Vt.StringArray(list(raw.names)))
        mesh.CreateExtentAttr(
            vec3f(np.stack([clip.reshape(-1, 3).min(axis=0), clip.reshape(-1, 3).max(axis=0)]))
        )
        stage.GetRootLayer().Save()


def front_vertex(raw: Raw, rest: FloatArray) -> int:
    """The body vertex furthest along the heading, on the head when the body has head regions."""
    skin = groom.Skin(rest, raw.triangles, raw.corner_st)
    body = coat.main_component(skin)
    head = np.isin(raw.regions, [raw.names.index(r) for r in HEAD_REGIONS if r in raw.names])
    on_head = np.zeros(len(rest), dtype=bool)
    on_head[raw.triangles[head]] = True
    candidates = np.flatnonzero(body & on_head) if (body & on_head).any() else np.flatnonzero(body)
    return int(candidates[np.argmax(rest[candidates, 2])])


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return round(float(rng.uniform(lo, hi)), 6)


def _fur(
    name: str,
    regions: tuple[str, ...],
    zone: tuple[float, float, float],
    *,
    lift: float,
    droop: float,
    length_m: float,
    width_m: tuple[float, float],
    frizz_m: float,
    clump: tuple[int, float],
    density: float,
) -> Json:
    """A fur layer template; `density` is hairs per square metre, resolved to a count against the skin."""
    return {
        "name": name,
        "regions": list(regions),
        "zone": {"geodesic": [zone[0], zone[1]], "min_up": zone[2]},
        "lift": lift,
        "droop": droop,
        "density": density,
        "groom": {
            "width_root_m": width_m[0],
            "width_tip_m": width_m[1],
            "length_m": length_m,
            "length_jitter": 0.25,
            "frizz_m": frizz_m,
            "neighbours": 4,
            "clump_size": clump[0],
            "clump_strength": clump[1],
        },
    }


def _short_coat(rng: np.random.Generator, size: float) -> list[Json]:
    length = _u(rng, 0.008, 0.02) * size
    whole = (0.0, 1.0, -1.0)
    return [
        _fur("under", FURRED, whole, lift=_u(rng, 0.12, 0.25), droop=_u(rng, 0.0, 0.1), length_m=length,
             width_m=(0.00012, 0.00004), frizz_m=0.0003, clump=(25, _u(rng, 0.1, 0.35)), density=220_000),
        _fur("guard", FURRED, whole, lift=_u(rng, 0.1, 0.2), droop=_u(rng, 0.05, 0.15),
             length_m=length * _u(rng, 1.4, 1.9), width_m=(0.0002, 0.00005), frizz_m=0.0006,
             clump=(12, _u(rng, 0.2, 0.45)), density=40_000),
    ]  # fmt: skip


def _long_coat(rng: np.random.Generator, size: float) -> list[Json]:
    whole = (0.0, 1.0, -1.0)
    length = _u(rng, 0.04, 0.09) * size
    return [
        _fur("under", FURRED, whole, lift=_u(rng, 0.2, 0.35), droop=_u(rng, 0.05, 0.2),
             length_m=length * 0.4, width_m=(0.00012, 0.00004), frizz_m=0.0015,
             clump=(20, _u(rng, 0.1, 0.3)), density=160_000),
        _fur("guard", FURRED, whole, lift=_u(rng, 0.08, 0.2), droop=_u(rng, 0.3, 0.6), length_m=length,
             width_m=(0.00022, 0.00005), frizz_m=0.002, clump=(30, _u(rng, 0.4, 0.7)), density=50_000),
    ]  # fmt: skip


def _woolly(rng: np.random.Generator, size: float) -> list[Json]:
    return [
        _fur("wool", FURRED, (0.0, 1.0, -1.0), lift=_u(rng, 0.4, 0.6), droop=_u(rng, 0.1, 0.3),
             length_m=_u(rng, 0.04, 0.07) * size, width_m=(0.00018, 0.00006), frizz_m=_u(rng, 0.004, 0.007),
             clump=(40, _u(rng, 0.55, 0.8)), density=150_000),
    ]  # fmt: skip


def _mane(rng: np.random.Generator, size: float) -> list[Json]:
    return [
        _fur("mane", ("head", "body"), (_u(rng, 0.06, 0.12), _u(rng, 0.25, 0.38), -0.35),
             lift=_u(rng, 0.25, 0.4), droop=_u(rng, 0.4, 0.8), length_m=_u(rng, 0.12, 0.25) * size,
             width_m=(0.00025, 0.00006), frizz_m=0.004, clump=(30, _u(rng, 0.5, 0.75)), density=90_000),
    ]  # fmt: skip


def _whiskers(rng: np.random.Generator, size: float) -> list[Json]:
    return [
        _fur("whiskers", ("head", "jaw", "head_detail", "body"), (0.0, _u(rng, 0.03, 0.05), -0.6),
             lift=_u(rng, 0.7, 0.85), droop=_u(rng, 0.1, 0.25), length_m=_u(rng, 0.05, 0.11) * size,
             width_m=(0.0004, 0.00005), frizz_m=0.0, clump=(0, 0.0), density=12_000),
    ]  # fmt: skip


def _down(rng: np.random.Generator, size: float) -> list[Json]:
    return [
        _fur("down", ("head", "body", "tail", "leg", "jaw", "head_detail"), (0.0, 1.0, -1.0),
             lift=_u(rng, 0.5, 0.7), droop=_u(rng, 0.0, 0.1), length_m=_u(rng, 0.005, 0.012) * size,
             width_m=(0.0001, 0.00003), frizz_m=0.0015, clump=(15, _u(rng, 0.2, 0.5)), density=150_000),
    ]  # fmt: skip


FUR_TEMPLATES: dict[str, Callable[[np.random.Generator, float], list[Json]]] = {
    "short_coat": _short_coat,
    "long_coat": _long_coat,
    "woolly": _woolly,
    "mane": _mane,
    "whiskers": _whiskers,
    "down": _down,
}


def _contour_feathers(rng: np.random.Generator, size: float) -> list[Json]:
    length = _u(rng, 0.025, 0.05) * size
    return [
        {
            "name": "contour",
            "regions": ["head", "body", "tail", "jaw", "head_detail"],
            "zone": {"geodesic": [_u(rng, 0.04, 0.08), 1.0], "min_up": -0.6},
            "density": 9_000.0,
            "params": {
                "length_m": length,
                "length_jitter": 0.2,
                "vane_width_m": length * _u(rng, 0.22, 0.32),
                "barbs": 18,
                "barb_angle_deg": _u(rng, 35.0, 55.0),
                "lift": _u(rng, 0.15, 0.3),
                "curl": _u(rng, 0.2, 0.4),
                "shaft_width_m": 0.0005,
                "barb_width_m": 0.00012,
            },
        }
    ]


FEATHER_TEMPLATES: dict[str, Callable[[np.random.Generator, float], list[Json]]] = {
    "contour_feathers": _contour_feathers,
}
COAT_CLASSES = (*FUR_TEMPLATES, *FEATHER_TEMPLATES, "bare")


def _area(skin: groom.Skin) -> float:
    tri = skin.points[skin.triangles]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def _zone(raw: Json) -> coat.Zone:
    lo, hi = cast("list[float]", raw["geodesic"])
    return coat.Zone((lo, hi), float(cast("float", raw["min_up"])))


def _resolve_fur(layer: Json, field: coat.Field, budget: float) -> Json:
    part, _ = coat.select(
        field, tuple(cast("list[str]", layer["regions"])), _zone(cast("Json", layer["zone"]))
    )
    count = max(16, round(float(cast("float", layer.pop("density"))) * _area(part) * budget))
    params = cast("Json", layer["groom"])
    size = int(cast("int", params.pop("clump_size")))
    params["count"] = count
    params["clump_count"] = count // size if size else 0
    thicken = min(MAX_THICKEN, budget**-0.5)
    for width in ("width_root_m", "width_tip_m"):
        params[width] = round(float(cast("float", params[width])) * thicken, 7)
    layer["guides"] = int(min(6000, max(64, count // 8)))
    return layer


def _resolve_feathers(layer: Json, field: coat.Field) -> Json:
    part, _ = coat.select(
        field, tuple(cast("list[str]", layer["regions"])), _zone(cast("Json", layer["zone"]))
    )
    params = cast("Json", layer["params"])
    params["count"] = max(40, round(float(cast("float", layer.pop("density"))) * _area(part)))
    return layer


def sample_coat(classes: list[str], field: coat.Field, rest: FloatArray, front: int, key: str) -> Json:
    """Concrete coat layers for hero.json, sized from the creature's skin and capped in total hairs.

    When the cap thins a coat, its hairs thicken by up to MAX_THICKEN to keep its coverage.
    """
    rng = stream(0, "creature-coat", key)
    size = float(np.clip((rest[:, 2].max() - rest[:, 2].min()) / 1.5, 0.35, 1.6))
    fur = [layer for c in classes if c in FUR_TEMPLATES for layer in FUR_TEMPLATES[c](rng, size)]
    plumage = [layer for c in classes if c in FEATHER_TEMPLATES for layer in FEATHER_TEMPLATES[c](rng, size)]
    unknown = [c for c in classes if c not in COAT_CLASSES]
    if unknown:
        raise ValueError(f"unknown coat classes {unknown}")
    wanted = sum(
        float(cast("float", layer["density"]))
        * _area(
            coat.select(
                field, tuple(cast("list[str]", layer["regions"])), _zone(cast("Json", layer["zone"]))
            )[0]
        )
        for layer in fur
    )
    budget = min(1.0, MAX_HAIRS / wanted) if wanted else 1.0
    morph = {
        "length": [_u(rng, 0.55, 0.8), 1.0],
        "density": [_u(rng, 0.5, 0.8), 1.0],
        "period_s": 8.0,
        "key_step": 48,
    }
    return {
        "morph": morph,
        "front_vertex": front,
        "segments": SEGMENTS,
        "animated_density": 0.4,
        "fur": [_resolve_fur(layer, field, budget) for layer in fur],
        "feathers": [_resolve_feathers(layer, field) for layer in plumage],
    }


def run_blender(command: list[str], script: str, job: Json, scratch: Path) -> None:
    listing = scratch / "job.json"
    files.write_text(listing, json.dumps(job, indent=1, sort_keys=True))
    subprocess.run([*command, str(BLENDER_DIR / script), "--", str(listing)], check=True, timeout=7200)


@dataclass(frozen=True)
class Tools:
    infinigen: list[str]
    blender: list[str]


def _export(recipe: Json, tools: Tools, raw_dir: Path) -> None:
    """Run the Blender export unless raw_dir already holds one for the same frames."""
    frames = sorted(
        {
            start + i
            for p in cast("dict[str, Json]", recipe["poses"]).values()
            for start in (int(cast("int", p["start"])),)
            for i in range(int(cast("int", p.get("frames", 1))))
        }
    )
    done = raw_dir / "creature.json"
    if (raw_dir / "skin.npz").exists() and json.loads(done.read_text())["frames"] == frames:
        return
    job = {**cast("Json", recipe["job"]), "frames": frames, "output": str(raw_dir)}
    if recipe["source"] == "infinigen":
        run_blender(tools.infinigen, "infinigen_creature.py", job, raw_dir.parent)
    else:
        run_blender(
            [*tools.blender, "-b", "--factory-startup", "--python"], "gltf_creature.py", job, raw_dir.parent
        )


def build_hero(recipe: Json, raw_dir: Path, folder: Path) -> Json:
    """Write a hero folder (poses, albedo, hero.json) from a raw export; returns the manifest."""
    key = str(recipe["key"])
    raw = read_raw(raw_dir)
    forward = cast("list[float] | None", recipe.get("forward"))
    axis = None if forward is None else (forward[0], forward[1], forward[2])
    oriented = orient(raw, axis, float(cast("float", recipe["length_m"])))
    index = {frame: i for i, frame in enumerate(raw.frames)}
    poses: dict[str, str] = {}
    first: FloatArray | None = None
    for name, pose in sorted(cast("dict[str, Json]", recipe["poses"]).items()):
        start, count = int(cast("int", pose["start"])), int(cast("int", pose.get("frames", 1)))
        clip = oriented[[index[start + i] for i in range(count)]]
        clip = in_place(clip) if count > 1 else clip
        if pose.get("loop"):
            clip = close_loop(clip, loop_period(clip, int(cast("int", pose.get("shortest", 8)))))
        write_pose(folder / "poses" / f"{name}.usdc", raw, clip)
        poses[name] = f"poses/{name}.usdc"
        first = clip[0] if first is None else first
    if first is None:
        raise ValueError(f"{key} has no poses")
    with files.replacing(folder / "textures" / "albedo.png") as temporary:
        shutil.copyfile(raw_dir / "albedo.png", temporary)
    skin = groom.Skin(first, raw.triangles, raw.corner_st)
    front = front_vertex(raw, first)
    field = coat.field(skin, raw.regions, raw.names, front)
    classes = cast("list[str]", recipe["coat"])
    provenance = cast("Json", recipe["provenance"])
    manifest: Json = {
        "key": key,
        "poses": poses,
        "skin_prim": SKIN_PRIM,
        "guide_prims": [],
        "albedo": "textures/albedo.png",
        "coat": sample_coat(classes, field, first, front, key),
        "coat_classes": classes,
        "length_m": recipe["length_m"],
        "generator": raw.info,
        "source": provenance["source"],
        "license": provenance["license"],
        "redistributable": provenance["redistributable"],
    }
    files.write_text(folder / "hero.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def prepare(recipes: list[Json], tools: Tools, library: Path) -> list[Json]:
    out: list[Json] = []
    for recipe in recipes:
        folder = library / str(recipe["key"])
        raw_dir = folder / "_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        _export(recipe, tools, raw_dir)
        out.append(build_hero(recipe, raw_dir, folder))
    return out
