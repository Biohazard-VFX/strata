"""Creature heroes over long shots: loop cyclic pose clips, move them at their stride and keep feet down.

A looping clip is authored in place, heading +Z. Its gait comes from the skin's lowest points: in
stance they sit on the ground and slide back at the ground speed, so the root advances by that
speed frame by frame and each stance frame lifts the clip so its lowest points meet the terrain.
Frames without a stance (a gallop's flight) interpolate both, keeping the authored rise. Clips
with no stance (idling, flapping) keep one lift; flapping ones fly clear of the ground.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom

from strata_datagen.fields import Ground
from strata_datagen.generators import hero, morph
from strata_datagen.rng import stream
from strata_datagen.timeline import FPS
from strata_datagen.usdutil import FloatArray, IntArray, new_stage

Json = dict[str, object]
COAT_LAYER = "hero_coat.usdc"
SKIN = "/hero/skin"
CONTACT_SHARE = 0.03
STANCE_RISE = 0.5
MIN_STANCE_SHARE = 0.3
MIN_STRIDE_MPS = 0.05
FLY_SWING = 0.25
HOVER = (0.3, 2.0)
LOWEST = 64


@dataclass(frozen=True)
class Gait:
    """A looping clip's cycle (frames, points, 3), ground speed per cycle frame and lift per cycle frame.

    `speed` is zero everywhere for clips without a stance; `flying` clips swing their lowest point
    by more than FLY_SWING of their height without a stance (wing beats) and are lifted clear.
    """

    cycle: FloatArray
    speed: FloatArray
    lift: FloatArray
    stance: NDArray[np.bool_]
    flying: bool

    @property
    def grounded(self) -> bool:
        return bool(self.stance.any())

    @property
    def speed_mps(self) -> float:
        return float(self.speed.mean())


def _cyclic(frames: IntArray, known: IntArray, values: FloatArray) -> FloatArray:
    period = len(frames)
    return np.interp(frames, np.concatenate([known - period, known, known + period]), np.tile(values, 3))


def _cycle(asset: Path) -> FloatArray | None:
    stage = Usd.Stage.Open(str(asset))
    points = UsdGeom.Mesh(stage.GetPrimAtPath(SKIN)).GetPointsAttr()
    times = points.GetTimeSamples()
    if not times:
        return None
    return np.stack([np.asarray(points.Get(t), dtype=np.float64) for t in times])


def gait(cycle: FloatArray) -> Gait:
    """The gait of a closed cycle (frames, points, 3) whose last frame equals its first.

    A frame is in stance when its lowest points barely rise or fall; their backward speed is the
    ground speed in that frame, interpolated through frames without a stance; their lift is
    interpolated too but never lets the clip's lowest point dip below the ground.
    """
    low = cycle[:-1, :, 1].min(axis=1)
    height = float(np.ptp(cycle[..., 1]))
    back, rise = np.empty(len(low)), np.empty(len(low))
    for f in range(len(low)):
        contact = cycle[f, :, 1] <= low[f] + CONTACT_SHARE * height
        step = (cycle[f + 1, contact] - cycle[f, contact]) * FPS
        back[f], rise[f] = -float(np.median(step[:, 2])), float(np.median(np.abs(step[:, 1])))
    stance = rise < STANCE_RISE * height
    frames = np.arange(len(low))
    if stance.mean() < MIN_STANCE_SHARE or float(back[stance].mean()) < MIN_STRIDE_MPS:
        flying = float(np.ptp(low)) > FLY_SWING * height
        still = np.zeros(len(low), dtype=bool)
        return Gait(cycle, np.zeros(len(low)), np.full(len(low), -float(low.min())), still, flying=flying)
    known = frames[stance]
    speed = np.maximum(_cyclic(frames, known, back[stance]), 0.0)
    lift = np.maximum(_cyclic(frames, known, -low[stance]), -low)
    return Gait(cycle, speed, lift, stance, flying=False)


def clip_gait(asset: Path) -> Gait | None:
    """The gait of a pose file's looping skin, or None for a static pose."""
    cycle = _cycle(asset)
    return None if cycle is None else gait(cycle)


def _path(yaw_deg: FloatArray, steps: FloatArray) -> tuple[FloatArray, FloatArray]:
    angle = np.radians(yaw_deg)
    x, z = np.cumsum(steps * np.sin(angle)), np.cumsum(steps * np.cos(angle))
    middle = len(steps) // 2
    return x - x[middle], z - z[middle]


def ground(motion: hero.HeroMotion, walk: Gait, terrain: Ground, rng: np.random.Generator) -> hero.HeroMotion:
    """The motion re-timed to the gait and lifted so the clip's lowest points meet the terrain.

    Grounded clips advance by each frame's stance speed, so planted feet stay put, and in stance
    the lowest of LOWEST points per frame rests on the terrain under it; frames between stances
    keep their authored rise. Clips without a stance keep the drawn speed; flying ones hover a
    seeded height above it.
    """
    period = len(walk.lift)
    phase = np.arange(len(motion.positions)) % period
    steps = np.concatenate([[0.0], walk.speed[phase[:-1]] / FPS]) if walk.grounded else None
    x, z = (motion.positions[:, 0], motion.positions[:, 2]) if steps is None else _path(motion.yaw_deg, steps)
    hover = float(rng.uniform(*HOVER)) * float(np.ptp(walk.cycle[..., 1])) if walk.flying else 0.0
    base = terrain.height(x, z) + walk.lift[phase] + hover
    angle = np.radians(motion.yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    correction = np.empty(len(phase))
    for i, f in enumerate(phase):
        points = walk.cycle[f]
        lowest = points[np.argsort(points[:, 1])[:LOWEST]]
        wx = x[i] + c[i] * lowest[:, 0] + s[i] * lowest[:, 2]
        wz = z[i] - s[i] * lowest[:, 0] + c[i] * lowest[:, 2]
        correction[i] = float(np.max(terrain.height(wx, wz) - lowest[:, 1] - base[i]))
    if walk.grounded:
        stance = np.flatnonzero(walk.stance[phase])
        correction = np.maximum(np.interp(np.arange(len(phase)), stance, correction[stance]), correction)
    else:
        correction = np.full(len(phase), max(float(correction.max()), 0.0))
    positions = np.stack([x, base + correction, z], axis=-1)
    speed = float(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1).mean() * FPS)
    return hero.HeroMotion(positions, motion.yaw_deg, speed)


def animate(
    stage: Usd.Stage, hero_path: str, asset: Path, *, asset_path: str, frames: tuple[int, ...], seed: int
) -> Json:
    """Loop the pose's clip over the shot (handles included) and morph its coat; returns what was done.

    Keyed coat widths are large, so they go to a binary sublayer beside the shot layer.
    """
    layer = Sdf.Layer.FindOrOpen(str(asset))  # pyright: ignore[reportUnknownMemberType]
    record: Json = {}
    first, last = frames[0] - 1, frames[-1] + 1
    period = int(max(layer.ListTimeSamplesForPath("/hero/skin.points") or [0.0]))
    if period > 0:
        morph.author_loop(
            stage.GetPrimAtPath(hero_path), asset_path, "/hero", period=period, first=first, last=last
        )
        record["loop_frames"] = period
    groom = cast("Sdf.PrimSpec | None", layer.GetPrimAtPath("/hero/groom"))
    raw = None if groom is None else cast("Json | None", groom.customData.get("morph"))
    if groom is None or raw is None:
        return record
    widths = layer.GetAttributeAtPath("/hero/groom.widths").default
    counts = layer.GetAttributeAtPath("/hero/groom.curveVertexCounts").default
    base = np.asarray(widths, dtype=np.float64).reshape(len(counts), -1)
    length = cast("list[float]", raw["length"])
    density = cast("list[float]", raw["density"])
    params = morph.CoatMorph(
        morph.Range(length[0], length[1]),
        morph.Range(density[0], density[1]),
        float(cast("float", raw["period_s"])),
        int(cast("int", raw["key_step"])),
    )
    path = Path(stage.GetRootLayer().realPath).parent / COAT_LAYER
    coat = new_stage(path)
    record["coat"] = morph.author_coat(
        coat,
        f"{hero_path}/groom",
        base,
        morph=params,
        frames=(first, last),
        rng=stream(seed, "coat", "morph"),
    )
    coat.GetRootLayer().Save()
    stage.GetRootLayer().subLayerPaths.append(f"./{COAT_LAYER}")
    return record
