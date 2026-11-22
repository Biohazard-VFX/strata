"""MaterialX standard_surface materials authored as UsdShade networks that Karma renders.

Every shader node id and input is checked against the MaterialX standard library before authoring.
Each surface shader also exports roughness and metallic as Karma AOVs (`inputs:aov:<name>`).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import MaterialX
from pxr import Gf, Sdf, Usd, UsdShade

from strata_datagen.usdutil import FloatArray

VERSION = "materials-0.1.0"
AOVS = ("roughness", "metallic")

_SDF = {
    "float": Sdf.ValueTypeNames.Float,
    "color3": Sdf.ValueTypeNames.Color3f,
    "vector2": Sdf.ValueTypeNames.Float2,
    "vector3": Sdf.ValueTypeNames.Float3,
    "string": Sdf.ValueTypeNames.String,
    "boolean": Sdf.ValueTypeNames.Bool,
    "filename": Sdf.ValueTypeNames.Asset,
    "surfaceshader": Sdf.ValueTypeNames.Token,
}

Value = float | bool | str | Gf.Vec2f | Gf.Vec3f | Sdf.AssetPath
Link = tuple[UsdShade.Shader, str]


@dataclass(frozen=True)
class Textures:
    """Texture files as asset paths relative to the layer that authors the material.

    Each albedo layer is mixed over the albedo by node `albedo_mix<i>`, whose `mix` input is 0
    unless a caller time-samples it.
    """

    albedo: str
    roughness: str | None = None
    albedo_layers: tuple[str, ...] = ()
    normal: str | None = None
    opacity: str | None = None


@dataclass(frozen=True)
class Look:
    textures: Textures
    tile_m: float | None = None
    foliage: bool = False
    roughness: float = 0.6
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    macro_variation: float = 0.0
    subsurface: float = 0.0
    subsurface_radius: tuple[float, float, float] = (1.0, 1.0, 1.0)
    subsurface_scale_m: float = 0.0
    specular: float = 0.5


@cache
def _library() -> MaterialX.Document:
    doc = MaterialX.createDocument()
    MaterialX.loadLibraries(
        MaterialX.getDefaultDataLibraryFolders(), MaterialX.getDefaultDataSearchPath(), doc
    )
    return doc


def _input_types(node_id: str) -> dict[str, str]:
    nodedef = _library().getNodeDef(node_id)
    if nodedef is None:
        raise KeyError(f"MaterialX has no nodedef {node_id}")
    return {i.getName(): i.getType() for i in nodedef.getActiveInputs()}


class _Graph:
    def __init__(self, stage: Usd.Stage, material: UsdShade.Material) -> None:
        self.stage = stage
        self.material = material

    def node(
        self,
        name: str,
        node_id: str,
        out_type: str,
        values: Mapping[str, Value] | None = None,
        links: Mapping[str, Link] | None = None,
    ) -> UsdShade.Shader:
        types = _input_types(node_id)
        shader = UsdShade.Shader.Define(self.stage, self.material.GetPath().AppendChild(name))
        shader.CreateIdAttr(node_id)
        for key, value in (values or {}).items():
            shader.CreateInput(key, _SDF[types[key]]).Set(value)
        for key, (source, output) in (links or {}).items():
            shader.CreateInput(key, _SDF[types[key]]).ConnectToSource(source.ConnectableAPI(), output)
        shader.CreateOutput("out", _SDF[out_type])
        return shader


def _texcoord(graph: _Graph, tile_m: float | None) -> Link:
    st = graph.node("st", "ND_geompropvalue_vector2", "vector2", {"geomprop": "st"})
    if tile_m is None:
        return (st, "out")
    scaled = graph.node(
        "st_tiled", "ND_multiply_vector2FA", "vector2", {"in2": 1.0 / tile_m}, {"in1": (st, "out")}
    )
    return (scaled, "out")


def _image(graph: _Graph, name: str, kind: str, path: str, texcoord: Link) -> Link:
    shader = graph.node(name, f"ND_image_{kind}", kind, {"file": Sdf.AssetPath(path)}, {"texcoord": texcoord})
    shader.GetInput("file").GetAttr().SetColorSpace("srgb_texture" if kind == "color3" else "raw")
    return (shader, "out")


def _base_color(graph: _Graph, look: Look, texcoord: Link) -> Link:
    color = _image(graph, "albedo", "color3", look.textures.albedo, texcoord)
    for index, layer in enumerate(look.textures.albedo_layers):
        image = _image(graph, f"albedo_layer{index}", "color3", layer, texcoord)
        mixed = graph.node(
            f"albedo_mix{index}", "ND_mix_color3", "color3", {"mix": 0.0}, {"fg": image, "bg": color}
        )
        color = (mixed, "out")
    tinted = graph.node("tint", "ND_multiply_color3", "color3", {"in2": Gf.Vec3f(*look.tint)}, {"in1": color})
    if look.macro_variation <= 0.0:
        return (tinted, "out")
    macro_st = graph.node("macro_st", "ND_multiply_vector2FA", "vector2", {"in2": 0.07}, {"in1": texcoord})
    noise = graph.node(
        "macro_noise",
        "ND_noise2d_float",
        "float",
        {"amplitude": look.macro_variation, "pivot": 1.0},
        {"texcoord": (macro_st, "out")},
    )
    varied = graph.node(
        "macro", "ND_multiply_color3FA", "color3", None, {"in1": (tinted, "out"), "in2": (noise, "out")}
    )
    return (varied, "out")


def _roughness(graph: _Graph, look: Look, texcoord: Link) -> Link:
    if look.textures.roughness is None:
        constant = graph.node("roughness", "ND_constant_float", "float", {"value": look.roughness})
        return (constant, "out")
    return _image(graph, "roughness", "float", look.textures.roughness, texcoord)


def _surface_links(graph: _Graph, look: Look, texcoord: Link) -> dict[str, Link]:
    base = _base_color(graph, look, texcoord)
    links = {"base_color": base, "specular_roughness": _roughness(graph, look, texcoord)}
    if look.textures.normal is not None:
        tangent = _image(graph, "normal_map", "vector3", look.textures.normal, texcoord)
        normal = graph.node("normal", "ND_normalmap_float", "vector3", None, {"in": tangent})
        links["normal"] = (normal, "out")
    if look.textures.opacity is not None:
        alpha = _image(graph, "opacity_map", "float", look.textures.opacity, texcoord)
        opacity = graph.node("opacity", "ND_convert_float_color3", "color3", None, {"in": alpha})
        links["opacity"] = (opacity, "out")
    if look.foliage or look.subsurface > 0.0:
        links["subsurface_color"] = base
    return links


def _export_aovs(surface: UsdShade.Shader, roughness: Link, metallic: Link) -> None:
    for name, (source, output) in zip(AOVS, (roughness, metallic), strict=True):
        aov = surface.CreateInput(f"aov:{name}", Sdf.ValueTypeNames.Float)
        aov.ConnectToSource(source.ConnectableAPI(), output)


def author(stage: Usd.Stage, path: str, look: Look) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    graph = _Graph(stage, material)
    texcoord = _texcoord(graph, look.tile_m)
    links = _surface_links(graph, look, texcoord)
    metallic = graph.node("metallic", "ND_constant_float", "float", {"value": 0.0})
    links["metalness"] = (metallic, "out")
    values: dict[str, Value] = {"base": 1.0, "specular": look.specular}
    if look.foliage:
        values.update({"thin_walled": True, "subsurface": 0.35})
    if look.subsurface > 0.0:
        values.update(
            {
                "subsurface": look.subsurface,
                "subsurface_radius": Gf.Vec3f(*look.subsurface_radius),
                "subsurface_scale": look.subsurface_scale_m,
            }
        )
    surface = graph.node("surface", "ND_standard_surface_surfaceshader", "surfaceshader", values, links)
    _export_aovs(surface, links["specular_roughness"], links["metalness"])
    material.CreateSurfaceOutput("mtlx").ConnectToSource(surface.ConnectableAPI(), "out")
    return material


def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def animate_tint(material: UsdShade.Material, tints: FloatArray, first_frame: int) -> None:
    """Time-sample a look's albedo tint: row i is authored at first_frame - 1 + i."""
    tint = UsdShade.Shader(material.GetPrim().GetChild("tint")).GetInput("in2")
    for index, rgb in enumerate(tints):
        tint.Set(Gf.Vec3f(*(float(v) for v in rgb)), Usd.TimeCode(first_frame - 1 + index))
