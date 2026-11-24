"""UsdRender settings for Karma: beauty with AOVs and a deep (DCM) product, one file per frame."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pxr import Gf, Sdf, Tf, Usd, UsdRender

from strata_dataset import roles, sampling

VERSION = "render-settings-0.4.1"
SETTINGS_PATH = "/Render/settings"


@dataclass(frozen=True)
class Aov:
    name: str
    source_type: str
    source_name: str
    data_type: str
    file_format: str
    pixel_filter: str = '["ubox",{}]'


BEAUTY_AOVS = (
    Aov("C", "lpe", "C.*[LO]", "color4f", "color4h"),
    Aov("albedo", "lpe", "CDA", "color3f", "color3h"),
    Aov("N", "raw", "ray:hitN_camera", "normal3f", "half3"),
    Aov("roughness", "raw", "roughness", "float", "half"),
    Aov("metallic", "raw", "metallic", "float", "half"),
    Aov("Z", "raw", "ray:hitPz", "float", "float", '["minmax",{"mode":"zmin"}]'),
    Aov("instance_id", "primvar", "instance_id", "int", "float", '["minmax",{"mode":"zmin"}]'),
    Aov("instance_check", "primvar", "instance_check", "float", "float", '["minmax",{"mode":"zmin"}]'),
    Aov("instance_check2", "primvar", "instance_check2", "float", "float", '["minmax",{"mode":"zmin"}]'),
    Aov("denoised", "lpe", "C.*[LO]", "color4f", "color4h"),
    *(Aov(role, "primvar", role, "float", "float") for role in roles.ROLES),
    Aov("coverage", "primvar", "coverage", "float", "float"),
)
DENOISER = json.dumps(
    [["denoise", {"engine": "oidn", "use_albedo_input": True, "use_n_input": True, "aovs": ["denoised"]}]]
)
DEEP_AOVS = ("C", "instance_id", "instance_check", "instance_check2", "coverage", *roles.ROLES)
DEEP_COMPRESSION = 5
DEEP_Z_BIAS_M = 0.0


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    path_traced_samples: int
    frames: tuple[int, ...]
    motion_blur: bool
    depth_of_field: bool
    shot_seed: int


def _var(stage: Usd.Stage, aov: Aov) -> UsdRender.Var:
    var = UsdRender.Var.Define(stage, f"/Render/vars/{aov.name}")
    var.CreateSourceTypeAttr(aov.source_type)
    var.CreateSourceNameAttr(aov.source_name)
    var.CreateDataTypeAttr(aov.data_type)
    prim = var.GetPrim()
    for name, value in (
        ("driver:parameters:aov:name", aov.name),
        ("driver:parameters:aov:husk:name", aov.name),
        ("driver:parameters:aov:husk:channel_prefix", "" if aov.name == "C" else aov.name),
    ):
        prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)
    prim.CreateAttribute("driver:parameters:aov:karma:filter", Sdf.ValueTypeNames.String).Set(
        aov.pixel_filter
    )
    prim.CreateAttribute("driver:parameters:aov:husk:format", Sdf.ValueTypeNames.Token).Set(aov.file_format)
    prim.CreateAttribute("driver:parameters:aov:husk:multiSampled", Sdf.ValueTypeNames.Bool).Set(
        aov.source_type == "lpe"
    )
    return var


def _product(
    stage: Usd.Stage, name: str, folder: str, frames: tuple[int, ...], variables: list[Sdf.Path]
) -> UsdRender.Product:
    product = UsdRender.Product.Define(stage, f"/Render/products/{name}")
    attr = product.CreateProductNameAttr()
    for frame in frames:
        attr.Set(f"{folder}/{frame:04d}.exr", Usd.TimeCode(frame))
    product.CreateOrderedVarsRel().SetTargets(variables)
    product.GetPrim().CreateAttribute(
        "driver:parameters:husk:OpenEXR:compression", Sdf.ValueTypeNames.String
    ).Set("zips")
    return product


def _deep(stage: Usd.Stage, frames: tuple[int, ...], variables: list[Sdf.Path]) -> UsdRender.Product:
    deep = _product(stage, "deep", "deep", frames, variables)
    deep.CreateProductTypeAttr("karma:deep")
    prim = deep.GetPrim()
    prim.CreateAttribute("karma:product:dcmcompression", Sdf.ValueTypeNames.Int).Set(DEEP_COMPRESSION)
    prim.CreateAttribute("karma:product:dcmofsize", Sdf.ValueTypeNames.Int).Set(1)
    prim.CreateAttribute("karma:product:dcmzbias", Sdf.ValueTypeNames.Float).Set(DEEP_Z_BIAS_M)
    return deep


def _karma(settings: UsdRender.Settings, config: RenderConfig) -> None:
    prim = settings.GetPrim()
    for name, kind, value in (
        ("karma:global:pathtracedsamples", Sdf.ValueTypeNames.Int, config.path_traced_samples),
        ("karma:global:abortnogpudevice", Sdf.ValueTypeNames.Bool, True),
        ("karma:global:pixelfilter", Sdf.ValueTypeNames.String, "box"),
        ("karma:global:imagefilter", Sdf.ValueTypeNames.String, DENOISER),
        ("karma:global:pixelfilterscale", Sdf.ValueTypeNames.Float, 1.0),
        ("karma:object:xformsamples", Sdf.ValueTypeNames.Int, 3),
        ("karma:object:geosamples", Sdf.ValueTypeNames.Int, 1),
    ):
        prim.CreateAttribute(name, kind).Set(value)
    schedule = sampling.schedule(config.shot_seed, config.frames)
    seed = prim.CreateAttribute("karma:global:randomseed", Sdf.ValueTypeNames.Int)
    if schedule["mode"] == "fixed":
        seed.Set(schedule["seeds"][0])
    else:
        for frame, value in zip(config.frames, schedule["seeds"], strict=True):
            seed.Set(value, Usd.TimeCode(frame))


def sampling_errors(shot: Path, manifest: Mapping[str, object]) -> list[str]:
    render = cast("Mapping[str, object]", manifest["render"])
    if render.get("sampling_seed") != {"mode": "karma_default"}:
        return []
    scene = shot / "scene" / "shot.usda"
    if not scene.exists():
        return []
    try:
        stage = Usd.Stage.Open(str(scene))
        prim = stage.GetPrimAtPath(SETTINGS_PATH)
        if not prim:
            return ["render.sampling_seed karma_default has no USD render settings evidence"]
        attr = prim.GetAttribute("karma:global:randomseed")
        if attr and attr.HasAuthoredValueOpinion():
            return ["render.sampling_seed karma_default contradicts authored USD randomseed"]
    except (OSError, RuntimeError, Tf.ErrorException) as error:
        return [f"render.sampling_seed karma_default has unreadable USD evidence: {error}"]
    return []


def author(stage: Usd.Stage, camera: str, config: RenderConfig) -> UsdRender.Settings:
    variables = {aov.name: _var(stage, aov).GetPath() for aov in BEAUTY_AOVS}
    beauty = _product(stage, "beauty", "beauty", config.frames, list(variables.values()))
    deep = _deep(stage, config.frames, [variables[name] for name in DEEP_AOVS])
    settings = UsdRender.Settings.Define(stage, SETTINGS_PATH)
    settings.CreateCameraRel().SetTargets([Sdf.Path(camera)])
    settings.CreateResolutionAttr(Gf.Vec2i(config.width, config.height))
    settings.CreatePixelAspectRatioAttr(1.0)
    settings.CreateProductsRel().SetTargets([beauty.GetPath(), deep.GetPath()])
    settings.CreateDisableMotionBlurAttr(not config.motion_blur)
    settings.CreateDisableDepthOfFieldAttr(not config.depth_of_field)
    settings.CreateIncludedPurposesAttr(["default", "render"])
    _karma(settings, config)
    stage.SetMetadata("renderSettingsPrimPath", SETTINGS_PATH)
    return settings
