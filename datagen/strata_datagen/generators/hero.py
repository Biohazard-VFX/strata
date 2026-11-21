"""Place a prepared hero (skin plus groom) on the ground, static or moving across the shot."""

from dataclasses import dataclass

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

from strata_datagen.fields import Ground
from strata_datagen.usdutil import FloatArray

VERSION = "hero-0.1.0"
SKIN_ID = 2
GROOM_ID = 3


@dataclass(frozen=True)
class HeroMotion:
    """Root position (on the ground) and heading per sample: one per frame plus a handle each side."""

    positions: FloatArray
    yaw_deg: FloatArray
    speed_mps: float


def sample_motion(
    rng: np.random.Generator, ground: Ground, samples: int, foot_offset_m: float, speed_mps: float | None
) -> HeroMotion:
    drawn = float(rng.uniform(0.2, 1.0)) if rng.random() < 0.6 else 0.0
    speed = drawn if speed_mps is None else speed_mps
    heading = float(rng.uniform(0.0, 360.0))
    turn = float(rng.uniform(-15.0, 15.0))
    t = (np.arange(samples) - samples // 2) / 24.0
    yaw = heading + turn * t
    angle = np.radians(yaw)
    x, z = speed * t * np.sin(angle), speed * t * np.cos(angle)
    y = ground.height(x, z) + foot_offset_m
    return HeroMotion(np.stack([x, y, z], axis=-1), yaw, speed)


def _bind(stage: Usd.Stage, path: Sdf.Path, material: UsdShade.Material, instance_id: int) -> None:
    prim = stage.OverridePrim(path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "instance_id", Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant
    ).Set(instance_id)


def author(
    stage: Usd.Stage,
    path: str,
    asset: str,
    motion: HeroMotion,
    *,
    first_frame: int,
    skin: UsdShade.Material,
    groom: UsdShade.Material,
) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    xform.GetPrim().GetReferences().AddReference(asset)
    translate, rotate = xform.AddTranslateOp(), xform.AddRotateYOp()
    for index, (position, yaw) in enumerate(zip(motion.positions, motion.yaw_deg, strict=True)):
        time = Usd.TimeCode(first_frame - 1 + index)
        translate.Set((float(position[0]), float(position[1]), float(position[2])), time)
        rotate.Set(float(yaw), time)
    _bind(stage, xform.GetPath().AppendChild("skin"), skin, SKIN_ID)
    _bind(stage, xform.GetPath().AppendChild("groom"), groom, GROOM_ID)
    return xform
