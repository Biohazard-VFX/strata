"""UsdRender settings for Karma: beauty with AOVs and a deep (DCM) product, one file per frame."""

import json
from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdRender

VERSION = "render-settings-0.3.0"
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
    Aov("denoised", "lpe", "C.*[LO]", "color4f", "color4h"),
    Aov("foreground", "primvar", "foreground", "float", "float"),
    Aov("coverage", "primvar", "coverage", "float", "float"),
)
DENOISER = json.dumps(
    [["denoise", {"engine": "oidn", "use_albedo_input": True, "use_n_input": True, "aovs": ["denoised"]}]]
)
DEEP_AOVS = ("C", "instance_id", "coverage")
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
