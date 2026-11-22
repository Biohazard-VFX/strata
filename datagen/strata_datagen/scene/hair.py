"""Grow a shot's hairstyle on the hero and carry it with the hero's skin.

Static poses grow on the prepared skin. Animated clips grow on the skin's rest points (the clip's
`rest` primvar): a similarity transform fitted to the upper head at every clip sample carries the
hair rigidly, and at each hairstyle key every strand is shifted by its root's remaining offset, so
roots follow a morphing head shape between keys by linear interpolation.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import KDTree

from strata_datagen import regions as skin_regions
from strata_datagen.generators import hairstyle, hero, human, materials
from strata_datagen.library import heroes
from strata_datagen.library.catalogue import HeroHair
from strata_datagen.rng import stream
from strata_datagen.scene import looks
from strata_datagen.timeline import FPS
from strata_datagen.usdutil import FloatArray, IntArray, new_stage, tag_instances

LAYER = "hair.usdc"
NEIGHBOURS = 16


@dataclass(frozen=True)
class Binding:
    """Rest skin of an animated hero and the rigid head vertices fitted per sample."""

    rest: FloatArray
    triangles: IntArray
    fitted: IntArray


@dataclass(frozen=True)
class Follow:
    """Per clip sample: shot time code and the rest-to-posed matrix (row vectors, as USD)."""

    codes: FloatArray
    matrices: FloatArray


def rest_head(path: Path, hair: HeroHair) -> tuple[hairstyle.Head, Binding]:
    """The skin's rest points as the hairstyle head; fits use the upper half of scalp and face."""
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(hair.skin_prim))
    if hair.rest_primvar is None:
        raise ValueError(f"{path}: an animated hero needs a rest primvar to bind hair")
    rest = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar(hair.rest_primvar).Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    corners, _ = heroes.fan(counts)
    triangles = indices[corners]
    weights = skin_regions.read(mesh, hair.regions)
    head = weights["scalp"] >= 0.5
    if "brows" in weights:
        head |= weights["brows"] >= 0.5
    candidates = np.flatnonzero(head)
    fitted = candidates[rest[candidates, 1] >= np.median(rest[candidates, 1])]
    grown = hairstyle.Head(rest, triangles, weights, np.array([0.0, 1.0, 0.0]), np.array(hair.forward))
    return grown, Binding(rest, triangles, fitted)


def similarity(source: FloatArray, target: FloatArray) -> FloatArray:
    """The 4x4 row-vector matrix of the scaled rigid motion that best maps source onto target."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    u, sigma, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(sigma) @ sign) / (a**2).sum(axis=1).mean())
    out = np.eye(4)
    out[:3, :3] = scale * rotation.T
    out[3, :3] = mu_t - mu_s @ out[:3, :3]
    return out


def apply(matrix: FloatArray, points: FloatArray) -> FloatArray:
    return points @ matrix[:3, :3] + matrix[3, :3]


def unapply(matrix: FloatArray, points: FloatArray) -> FloatArray:
    return (points - matrix[3, :3]) @ np.linalg.inv(matrix[:3, :3])


def posed_points(path: Path, skin_prim: str, times: FloatArray) -> FloatArray:
    stage = Usd.Stage.Open(str(path))
    attribute = UsdGeom.Mesh(stage.GetPrimAtPath(skin_prim)).GetPointsAttr()
    return np.stack([np.asarray(attribute.Get(Usd.TimeCode(float(t))), dtype=np.float64) for t in times])


def clip_times(path: Path, skin_prim: str, low: float, high: float) -> FloatArray:
    stage = Usd.Stage.Open(str(path))
    attribute = UsdGeom.Mesh(stage.GetPrimAtPath(skin_prim)).GetPointsAttr()
    return np.asarray(attribute.GetTimeSamplesInInterval(Gf.Interval(low, high)), dtype=np.float64)


def follow(binding: Binding, posed: FloatArray) -> FloatArray:
    """Rest-to-posed similarity per posed sample, fitted on the binding's rigid head vertices."""
    source = binding.rest[binding.fitted]
    return np.stack([similarity(source, p[binding.fitted]) for p in posed])


def attach(binding: Binding, roots: FloatArray) -> tuple[IntArray, FloatArray]:
    """The rest triangle under each root and the root's barycentric weights on it."""
    tri = binding.rest[binding.triangles]
    _, near = KDTree(tri.mean(axis=1)).query(roots, k=NEIGHBOURS)
    candidates = np.asarray(near, dtype=np.int64)
    a, b, c = (tri[candidates, k] for k in range(3))
    e1, e2, d = b - a, c - a, roots[:, None] - a
    d11, d12, d22 = (e1 * e1).sum(-1), (e1 * e2).sum(-1), (e2 * e2).sum(-1)
    p1, p2 = (d * e1).sum(-1), (d * e2).sum(-1)
    det = np.maximum(d11 * d22 - d12 * d12, 1e-30)
    v, w = (d22 * p1 - d12 * p2) / det, (d11 * p2 - d12 * p1) / det
    bary = np.clip(np.stack([1.0 - v - w, v, w], axis=-1), 0.0, None)
    bary /= bary.sum(axis=-1, keepdims=True)
    closest: FloatArray = np.einsum("nck,nckd->ncd", bary, tri[candidates])
    best = np.argmin(np.linalg.norm(closest - roots[:, None], axis=-1), axis=1)
    rows = np.arange(len(roots))
    return candidates[rows, best], bary[rows, best]


def root_offsets(binding: Binding, roots: FloatArray, posed: FloatArray, matrix: FloatArray) -> FloatArray:
    """Each root's offset from rest left after undoing the fitted motion, on its rest triangle."""
    residual = unapply(matrix, posed) - binding.rest
    face, bary = attach(binding, roots)
    return np.einsum("nk,nkd->nd", bary, residual[binding.triangles[face]])


def carried(
    built: hairstyle.Hairstyle, binding: Binding, posed: FloatArray, matrices: FloatArray
) -> hairstyle.Hairstyle:
    """Shift each key's strands by their roots' residual offsets at that key."""
    sets: list[hairstyle.StrandSet] = []
    for strand_set in built.sets:
        roots = strand_set.curves[0, :, 0]
        shifts = np.stack([root_offsets(binding, roots, p, m) for p, m in zip(posed, matrices, strict=True)])
        sets.append(replace(strand_set, curves=strand_set.curves + shifts[:, :, None]))
    return replace(built, sets=tuple(sets))


def _author_follow(layer: Usd.Stage, motion: Follow) -> None:
    op = UsdGeom.Xformable(layer.GetPrimAtPath("/hair")).AddTransformOp()
    for code, matrix in zip(motion.codes, motion.matrices, strict=True):
        op.Set(Gf.Matrix4d(*matrix.ravel().tolist()), Usd.TimeCode(float(code)))


def author(
    stage: Usd.Stage,
    scene_dir: Path,
    pose_file: Path,
    hair: HeroHair,
    style: hairstyle.Style,
    *,
    frames: tuple[int, ...],
    placement: human.ClipPlacement | None,
    seed: int,
) -> None:
    """Write scene/hair.usdc under /World/hero/hair with keys every hairstyle.KEY_SECONDS."""
    first = frames[0]
    seconds = hairstyle.key_seconds((frames[-1] - first) / FPS)
    codes = np.array([first + s * FPS for s in seconds])
    rng = stream(seed, "hairstyle", "build")
    motion = None
    if placement is None:
        built = hairstyle.build(heroes.read_head(pose_file, hair.regions, hair.forward), style, rng, seconds)
    else:
        head, binding = rest_head(pose_file, hair)
        built = hairstyle.build(head, style, rng, seconds)
        offset = float(first - placement.start)
        times = clip_times(pose_file, hair.skin_prim, first - 1.5 - offset, frames[-1] + 1.5 - offset)
        motion = Follow(times + offset, follow(binding, posed_points(pose_file, hair.skin_prim, times)))
        at_keys = posed_points(pose_file, hair.skin_prim, codes - offset)
        built = carried(built, binding, at_keys, follow(binding, at_keys))
    layer = new_stage(scene_dir / LAYER)
    names = hairstyle.author(layer, "/hair", built, tuple(codes.tolist()) if len(codes) > 1 else ())
    if motion is not None:
        _author_follow(layer, motion)
    layer.SetDefaultPrim(layer.GetPrimAtPath("/hair"))
    layer.GetRootLayer().Save()
    stage.DefinePrim("/World/hero/hair").GetReferences().AddReference(f"./{LAYER}")
    keys = [(Usd.TimeCode(float(c)), s) for c, s in zip(codes, seconds, strict=True)]
    scalp, facial = looks.hair(stage, style, keys if len(keys) > 1 else [(Usd.TimeCode.Default(), 0.0)])
    for name in names:
        prim = stage.OverridePrim(f"/World/hero/hair/{name}")
        materials.bind(prim, scalp if name == "scalp" else facial)
        tag_instances(prim, np.array([hero.HAIR_ID]))
