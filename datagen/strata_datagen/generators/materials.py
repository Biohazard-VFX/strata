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
    """Texture files as asset paths relative to the layer that authors the material."""

    albedo: str
    roughness: str | None = None
    normal: str | None = None
    opacity: str | None = None


@dataclass(frozen=True)
class Look:
    """Without textures the surface is the constant base_color."""

    textures: Textures | None
    tile_m: float | None = None
    foliage: bool = False
    roughness: float = 0.6
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    macro_variation: float = 0.0
    base_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    metallic: float = 0.0


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


def _constant(graph: _Graph, name: str, kind: str, value: Value) -> Link:
    return (graph.node(name, f"ND_constant_{kind}", kind, {"value": value}), "out")


def _base_color(graph: _Graph, look: Look, textures: Textures, texcoord: Link) -> Link:
    color = _image(graph, "albedo", "color3", textures.albedo, texcoord)
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


def _roughness(graph: _Graph, look: Look, textures: Textures, texcoord: Link) -> Link:
    if textures.roughness is None:
        return _constant(graph, "roughness", "float", look.roughness)
    return _image(graph, "roughness", "float", textures.roughness, texcoord)


def _constant_links(graph: _Graph, look: Look) -> dict[str, Link]:
    return {
        "base_color": _constant(graph, "albedo", "color3", Gf.Vec3f(*look.base_color)),
        "specular_roughness": _constant(graph, "roughness", "float", look.roughness),
    }


def _surface_links(graph: _Graph, look: Look, textures: Textures) -> dict[str, Link]:
    texcoord = _texcoord(graph, look.tile_m)
    base = _base_color(graph, look, textures, texcoord)
    links = {"base_color": base, "specular_roughness": _roughness(graph, look, textures, texcoord)}
    if textures.normal is not None:
        tangent = _image(graph, "normal_map", "vector3", textures.normal, texcoord)
        normal = graph.node("normal", "ND_normalmap_float", "vector3", None, {"in": tangent})
        links["normal"] = (normal, "out")
    if textures.opacity is not None:
        alpha = _image(graph, "opacity_map", "float", textures.opacity, texcoord)
        opacity = graph.node("opacity", "ND_convert_float_color3", "color3", None, {"in": alpha})
        links["opacity"] = (opacity, "out")
    if look.foliage:
        links["subsurface_color"] = base
    return links


def _export_aovs(surface: UsdShade.Shader, roughness: Link, metallic: Link) -> None:
    for name, (source, output) in zip(AOVS, (roughness, metallic), strict=True):
        aov = surface.CreateInput(f"aov:{name}", Sdf.ValueTypeNames.Float)
        aov.ConnectToSource(source.ConnectableAPI(), output)


def author(stage: Usd.Stage, path: str, look: Look) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    graph = _Graph(stage, material)
    links = (
        _constant_links(graph, look) if look.textures is None else _surface_links(graph, look, look.textures)
    )
    links["metalness"] = _constant(graph, "metallic", "float", look.metallic)
    values: dict[str, Value] = {"base": 1.0, "specular": 0.5}
    if look.foliage:
        values.update({"thin_walled": True, "subsurface": 0.35})
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
