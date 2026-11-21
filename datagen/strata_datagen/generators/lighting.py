"""HDRI dome plus a matching distant sun, normalised to a target horizontal illuminance."""

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

VERSION = "lighting-0.1.0"
SUN_ANGLE_DEG = 0.53


@dataclass(frozen=True)
class Hdri:
    """A prepared latlong HDRI with its sun removed into separate, measured light.

    sun_direction is a unit vector in the dome's own frame (Karma dome: +Y up, image centre
    towards +Z). Irradiances are on a plane facing +Y (sky) and facing the sun (sun),
    in the same linear units as the image.
    """

    key: str
    asset: str
    sun_direction: tuple[float, float, float]
    sun_irradiance: float
    sun_color: tuple[float, float, float]
    sky_irradiance: float


@dataclass(frozen=True)
class Lighting:
    hdri: Hdri
    rotation_deg: float
    exposure_scale: float

    def sun_direction_world(self) -> tuple[float, float, float]:
        angle = np.radians(self.rotation_deg)
        x, y, z = self.hdri.sun_direction
        c, s = np.cos(angle), np.sin(angle)
        return (float(c * x + s * z), float(y), float(-s * x + c * z))


def sample(
    hdris: tuple[Hdri, ...],
    rng: np.random.Generator,
    target_irradiance: float,
    key: str | None,
    rotation_deg: float | None,
) -> Lighting:
    """Pick an HDRI and rotation (fixed values override draws) and normalise its total irradiance."""
    drawn = hdris[int(rng.integers(0, len(hdris)))]
    drawn_rotation = float(rng.uniform(0.0, 360.0))
    hdri = drawn if key is None else next(h for h in hdris if h.key == key)
    sun_up = hdri.sun_irradiance * max(hdri.sun_direction[1], 0.0)
    return Lighting(
        hdri=hdri,
        rotation_deg=drawn_rotation if rotation_deg is None else rotation_deg,
        exposure_scale=target_irradiance / max(hdri.sky_irradiance + sun_up, 1e-6),
    )


def _rotation_y(stage: Usd.Stage, path: str, degrees: float) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddRotateYOp().Set(degrees)
    return xform


def _sun_rotation(direction: tuple[float, float, float]) -> Gf.Vec3f:
    """Euler XYZ degrees that turn a distant light's +Z axis (it shines along -Z) to `direction`."""
    x, y, z = direction
    pitch = -np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(x, z))
    return Gf.Vec3f(float(pitch), float(yaw), 0.0)


def author(stage: Usd.Stage, path: str, lighting: Lighting) -> None:
    rig = _rotation_y(stage, path, lighting.rotation_deg)
    dome = UsdLux.DomeLight.Define(stage, rig.GetPath().AppendChild("dome"))
    dome.CreateTextureFileAttr(Sdf.AssetPath(lighting.hdri.asset))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(lighting.exposure_scale)
    dome.GetPrim().CreateAttribute("inputs:karma:light:renderlightgeo", Sdf.ValueTypeNames.Bool).Set(
        value=True
    )
    sun = UsdLux.DistantLight.Define(stage, rig.GetPath().AppendChild("sun"))
    sun.CreateAngleAttr(SUN_ANGLE_DEG)
    sun.CreateIntensityAttr(lighting.exposure_scale * lighting.hdri.sun_irradiance)
    sun.CreateColorAttr(Gf.Vec3f(*lighting.hdri.sun_color))
    sun.CreateNormalizeAttr(defaultValue=True)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(_sun_rotation(lighting.hdri.sun_direction))
