"""HDRI dome plus a matching distant sun, normalised to a target horizontal illuminance."""

from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from strata_datagen import timeline
from strata_datagen.usdutil import FloatArray, IntArray, ints, tag_instances, vec2f, vec3f

VERSION = "lighting-0.3.0"
SUN_ANGLE_DEG = 0.53
BACKDROP_M = 3000.0
SKY_ID = 0


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
    """drift holds, per frame sample, a rotation offset in degrees and an exposure offset in stops."""

    hdri: Hdri
    rotation_deg: float
    exposure_scale: float
    drift: FloatArray = field(default_factory=lambda: np.zeros((1, 2)))

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
    *,
    samples: int,
) -> Lighting:
    """Pick an HDRI and rotation (fixed values override draws) and normalise its total irradiance."""
    drawn = hdris[int(rng.integers(0, len(hdris)))]
    drawn_rotation = float(rng.uniform(0.0, 360.0))
    hdri = drawn if key is None else next(h for h in hdris if h.key == key)
    sun_up = hdri.sun_irradiance * max(hdri.sun_direction[1], 0.0)
    t = timeline.sample_times(samples)
    duration = float(t[-1]) + 1.0
    turn = timeline.curve(rng, duration, 10.0, -25.0, 25.0)(t)
    stops = timeline.curve(rng, duration, 8.0, -0.4, 0.4)(t)
    return Lighting(
        hdri=hdri,
        rotation_deg=drawn_rotation if rotation_deg is None else rotation_deg,
        exposure_scale=target_irradiance / max(hdri.sky_irradiance + sun_up, 1e-6),
        drift=np.stack([turn, stops], axis=-1),
    )


def _sun_rotation(direction: tuple[float, float, float]) -> Gf.Vec3f:
    """Euler XYZ degrees that turn a distant light's +Z axis (it shines along -Z) to `direction`."""
    x, y, z = direction
    pitch = -np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(x, z))
    return Gf.Vec3f(float(pitch), float(yaw), 0.0)


def _backdrop_mesh(stage: Usd.Stage, path: Sdf.Path, segments: int = 128) -> UsdGeom.Mesh:
    """An inward-facing sphere whose st maps the HDRI exactly as Karma's latlong dome does."""
    u = np.linspace(0.0, 1.0, segments + 1)
    v = np.linspace(0.0, 1.0, segments // 2 + 1)
    uu, vv = np.meshgrid(u, v)
    theta, phi = np.pi * (1.0 - vv), 2.0 * np.pi * uu
    points = BACKDROP_M * np.stack(
        [np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)], axis=-1
    )
    columns = segments + 1
    rows = segments // 2 + 1
    index: IntArray = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    quads = np.stack([index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], axis=-1).reshape(-1, 4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(vec3f(points.reshape(-1, 3)))
    mesh.CreateFaceVertexCountsAttr(ints(np.full(len(quads), 4)))
    mesh.CreateFaceVertexIndicesAttr(ints(quads.ravel()))
    mesh.CreateDoubleSidedAttr(defaultValue=True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(vec2f(np.stack([uu, vv], axis=-1).reshape(-1, 2)))
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    return mesh


def _backdrop(stage: Usd.Stage, rig: Sdf.Path, lighting: Lighting) -> Usd.Attribute:
    """Camera-only sky geometry, so every pixel has a deep sample; returns its emission input."""
    mesh = _backdrop_mesh(stage, rig.AppendChild("backdrop"))
    prim = mesh.GetPrim()
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "karma:object:rendervisibility", Sdf.ValueTypeNames.String, UsdGeom.Tokens.constant
    ).Set("primary")
    tag_instances(prim, np.array([SKY_ID]))
    UsdGeom.PrimvarsAPI(prim).GetPrimvar("foreground").Set(0.0)
    material = UsdShade.Material.Define(stage, rig.AppendChild("backdrop_look"))
    image = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("sky"))
    image.CreateIdAttr("ND_image_color3")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(lighting.hdri.asset))
    image.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
    surface = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("surface"))
    surface.CreateIdAttr("ND_standard_surface_surfaceshader")
    surface.CreateInput("base", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        image.ConnectableAPI(), "out"
    )
    emission = surface.CreateInput("emission", Sdf.ValueTypeNames.Float)
    surface.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(surface.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return emission.GetAttr()


def author(stage: Usd.Stage, path: str, lighting: Lighting, first_frame: int) -> None:
    """Dome and sun under one rotated rig; drift sample i is authored at first_frame - 1 + i."""
    rig = UsdGeom.Xform.Define(stage, path)
    turn = rig.AddRotateYOp()
    dome = UsdLux.DomeLight.Define(stage, rig.GetPath().AppendChild("dome"))
    dome.CreateTextureFileAttr(Sdf.AssetPath(lighting.hdri.asset))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.GetPrim().CreateAttribute("inputs:karma:light:renderlightgeo", Sdf.ValueTypeNames.Bool).Set(
        value=True
    )
    sun = UsdLux.DistantLight.Define(stage, rig.GetPath().AppendChild("sun"))
    sun.CreateAngleAttr(SUN_ANGLE_DEG)
    sun.CreateColorAttr(Gf.Vec3f(*lighting.hdri.sun_color))
    sun.CreateNormalizeAttr(defaultValue=True)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(_sun_rotation(lighting.hdri.sun_direction))
    dome_intensity, sun_intensity = dome.CreateIntensityAttr(), sun.CreateIntensityAttr()
    sky = _backdrop(stage, rig.GetPath(), lighting)
    for index, (offset, stops) in enumerate(lighting.drift):
        time = Usd.TimeCode(first_frame - 1 + index) if len(lighting.drift) > 1 else Usd.TimeCode.Default()
        scale = lighting.exposure_scale * float(2.0**stops)
        turn.Set(lighting.rotation_deg + float(offset), time)
        dome_intensity.Set(scale, time)
        sky.Set(scale, time)
        sun_intensity.Set(scale * lighting.hdri.sun_irradiance, time)
