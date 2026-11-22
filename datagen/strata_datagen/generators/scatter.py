"""Instance Megascans plants, rocks and debris over the ground by mask-weighted density."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom, UsdShade

from strata_datagen import timeline
from strata_datagen.fields import Ground
from strata_datagen.usdutil import FloatArray, ints, quath, tag_instances, vec3f

VERSION = "scatter-0.3.0"
GROWTH_KEY_FRAMES = 24
GROWTH_FADE = 0.15
NATIVE_SCALE = (0.5, 2.0)


@dataclass(frozen=True)
class Prototype:
    """A model to instance: footprint is its largest horizontal extent, in metres at scale 1."""

    key: str
    asset: str
    material: str
    footprint_m: float
    height_m: float


@dataclass(frozen=True)
class Layer:
    """One instancer: prototypes, where they may grow, and how densely.

    Instances fill a disk of radius_m around centre, accepted with probability equal to the
    product of the named ground masks. Each instance's footprint is drawn log-uniformly from the
    part of footprint_range_m its prototype reaches within NATIVE_SCALE of its native size, and it
    sinks by sink_fraction of its height.
    """

    name: str
    prototypes: tuple[Prototype, ...]
    density_per_m2: float
    radius_m: float
    centre: tuple[float, float]
    masks: tuple[str, ...]
    footprint_range_m: tuple[float, float]
    align_to_ground: float
    sink_fraction: float


@dataclass(frozen=True)
class Disk:
    x: float
    z: float
    radius_m: float


@dataclass(frozen=True)
class Instances:
    positions: FloatArray
    orientations: FloatArray
    scales: FloatArray
    prototype_indices: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def fits(footprint_m: float, footprint_range_m: tuple[float, float]) -> bool:
    """Whether a model reaches the footprint range without leaving NATIVE_SCALE of its native size."""
    low, high = footprint_range_m
    return footprint_m * NATIVE_SCALE[0] <= high and footprint_m * NATIVE_SCALE[1] >= low


def _candidates(layer: Layer, rng: np.random.Generator) -> FloatArray:
    area = np.pi * layer.radius_m**2
    count = rng.poisson(layer.density_per_m2 * area)
    radius = layer.radius_m * np.sqrt(rng.random(count))
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    return np.stack(
        [layer.centre[0] + radius * np.cos(angle), layer.centre[1] + radius * np.sin(angle)], axis=-1
    )


def _accept(
    ground: Ground, layer: Layer, xz: FloatArray, exclusions: Sequence[Disk], rng: np.random.Generator
) -> NDArray[np.bool_]:
    weight: FloatArray = np.ones(xz.shape[0])
    for name in layer.masks:
        weight *= ground.mask(name, xz[:, 0], xz[:, 1])
    draws: FloatArray = rng.random(len(xz))
    keep: NDArray[np.bool_] = np.less(draws, weight)
    inside = np.abs(xz).max(axis=1) < ground.half_size_m * 0.98
    for disk in exclusions:
        keep &= np.hypot(xz[:, 0] - disk.x, xz[:, 1] - disk.z) > disk.radius_m
    return keep & inside


def _quaternions(yaw: FloatArray, normals: FloatArray, align: float) -> FloatArray:
    """Yaw about +Y, then tilt +Y towards the (partially applied) ground normal; rows are (w, x, y, z)."""
    up = np.array([0.0, 1.0, 0.0])
    target = (1.0 - align) * up + align * normals
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    axis = np.cross(up, target)
    sin = np.linalg.norm(axis, axis=1)
    cos = target[:, 1]
    tilt_angle = np.arctan2(sin, cos)
    axis = axis / np.maximum(sin, 1e-9)[:, None]
    tilt = np.concatenate([np.cos(tilt_angle / 2)[:, None], axis * np.sin(tilt_angle / 2)[:, None]], 1)
    spin = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.sin(yaw / 2), np.zeros_like(yaw)], 1)
    return _multiply(tilt, spin)


def _multiply(a: FloatArray, b: FloatArray) -> FloatArray:
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def place(ground: Ground, layer: Layer, exclusions: Sequence[Disk], rng: np.random.Generator) -> Instances:
    xz = _candidates(layer, rng)
    xz = xz[_accept(ground, layer, xz, exclusions, rng)]
    count = xz.shape[0]
    proto = rng.integers(0, len(layer.prototypes), count)
    footprint = np.array([p.footprint_m for p in layer.prototypes])[proto]
    height = np.array([p.height_m for p in layer.prototypes])[proto]
    if not all(fits(p.footprint_m, layer.footprint_range_m) for p in layer.prototypes):
        raise ValueError(f"{layer.name}: a prototype cannot reach {layer.footprint_range_m} m")
    low = np.log(np.maximum(layer.footprint_range_m[0], footprint * NATIVE_SCALE[0]))
    high = np.log(np.minimum(layer.footprint_range_m[1], footprint * NATIVE_SCALE[1]))
    scale = np.exp(rng.uniform(low, high, count)) / footprint
    y = ground.height(xz[:, 0], xz[:, 1]) - layer.sink_fraction * height * scale
    normals = ground.normal(xz[:, 0], xz[:, 1]).reshape(-1, 3)
    yaw = rng.uniform(0.0, 2.0 * np.pi, count)
    return Instances(
        positions=np.stack([xz[:, 0], y, xz[:, 1]], axis=-1),
        orientations=_quaternions(yaw, normals, layer.align_to_ground),
        scales=np.repeat(scale[:, None], 3, axis=1),
        prototype_indices=proto,
    )


def growth(rng: np.random.Generator, count: int, samples: int) -> FloatArray:
    """Scale multipliers (keys every GROWTH_KEY_FRAMES samples, instances) for a slowly drifting density.

    Each instance has a threshold; as the visible share of the layer rises and falls between 0.55
    and 1, instances grow from or shrink to nothing over a GROWTH_FADE band, never popping.
    """
    keys = np.arange(0, samples + GROWTH_KEY_FRAMES - 1, GROWTH_KEY_FRAMES)
    t = keys / timeline.FPS
    share = timeline.curve(rng, float(t[-1]) + 1.0, 8.0, 0.55, 1.0)(t)
    threshold = rng.random(count) * (1.0 - GROWTH_FADE)
    x = np.clip((share[:, None] - threshold[None]) / GROWTH_FADE, 0.0, 1.0)
    return np.maximum(x * x * (3.0 - 2.0 * x), 1e-3)


def author(
    stage: Usd.Stage,
    path: str,
    layer: Layer,
    instances: Instances,
    first_id: int,
    *,
    grow: tuple[FloatArray, int] | None = None,
) -> UsdGeom.PointInstancer:
    """A PointInstancer whose instances carry a unique integer `instance_id` primvar.

    grow is (growth(...) multipliers, first frame): scales are keyed every GROWTH_KEY_FRAMES.
    """
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    protos = instancer.GetPrim().GetPath().AppendChild("prototypes")
    UsdGeom.Scope.Define(stage, protos)
    targets: list[Sdf.Path] = []
    for index, proto in enumerate(layer.prototypes):
        prim = stage.DefinePrim(protos.AppendChild(f"p{index}"))
        prim.GetReferences().AddReference(proto.asset)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(proto.material)),
            UsdShade.Tokens.strongerThanDescendants,
        )
        targets.append(prim.GetPath())
    instancer.CreatePrototypesRel().SetTargets(targets)
    instancer.CreatePositionsAttr(vec3f(instances.positions))
    instancer.CreateOrientationsAttr(quath(instances.orientations))
    scales = instancer.CreateScalesAttr()
    if grow is None:
        scales.Set(vec3f(instances.scales))
    else:
        multipliers, first_frame = grow
        for key, row in enumerate(multipliers):
            time = Usd.TimeCode(first_frame - 1 + key * GROWTH_KEY_FRAMES)
            scales.Set(vec3f(instances.scales * row[:, None]), time)
    instancer.CreateProtoIndicesAttr(ints(instances.prototype_indices))
    tag_instances(instancer.GetPrim(), np.arange(first_id, first_id + len(instances)))
    return instancer
