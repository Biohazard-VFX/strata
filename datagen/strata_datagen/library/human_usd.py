"""Author a human hero as USD: animated subdivision meshes with MaterialX looks and skin regions.

Each part is a mesh with time-sampled points, a bound look and a `rest` primvar; the skin carries
region GeomSubsets.

Layout (default prim /hero): /hero/body (skin), /hero/eyes, /hero/garment<N>, /hero/looks/<look>.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

from strata_datagen.generators import materials
from strata_datagen.usdutil import FloatArray, IntArray, ints, new_stage, vec2f, vec3f

ROOT = "/hero"
LOOKS = "/hero/looks"
REGION_FAMILY = "hero_region"
REST_PRIMVAR = "rest"


@dataclass(frozen=True)
class MeshPart:
    name: str
    counts: IntArray
    indices: IntArray
    uv: FloatArray
    rest: FloatArray
    look: str


def _mesh(stage: Usd.Stage, part: MeshPart) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/{part.name}")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    mesh.CreateDoubleSidedAttr(defaultValue=True)
    mesh.CreateFaceVertexCountsAttr(ints(part.counts))
    mesh.CreateFaceVertexIndicesAttr(ints(part.indices))
    primvars = UsdGeom.PrimvarsAPI(mesh)
    primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        vec2f(part.uv)
    )
    primvars.CreatePrimvar(REST_PRIMVAR, Sdf.ValueTypeNames.Point3fArray, UsdGeom.Tokens.vertex).Set(
        vec3f(part.rest)
    )
    return mesh


def _points(mesh: UsdGeom.Mesh, samples: FloatArray, times: FloatArray | None) -> None:
    points, extent = mesh.CreatePointsAttr(), mesh.CreateExtentAttr()
    if times is None:
        points.Set(vec3f(samples[0]))
        extent.Set(vec3f(np.stack([samples[0].min(axis=0), samples[0].max(axis=0)])))
        return
    for sample, time in zip(samples, times, strict=True):
        code = Usd.TimeCode(float(time))
        points.Set(vec3f(sample), code)
        extent.Set(vec3f(np.stack([sample.min(axis=0), sample.max(axis=0)])), code)


def _regions(mesh: UsdGeom.Mesh, regions: dict[str, IntArray]) -> None:
    for name, faces in sorted(regions.items()):
        UsdGeom.Subset.CreateGeomSubset(
            mesh, name, UsdGeom.Tokens.face, ints(faces), REGION_FAMILY, UsdGeom.Tokens.unrestricted
        )
    UsdGeom.Subset.SetFamilyType(mesh, REGION_FAMILY, UsdGeom.Tokens.unrestricted)


def write(
    path: Path,
    *,
    parts: list[MeshPart],
    samples: dict[str, FloatArray],
    times: FloatArray | None,
    looks: dict[str, materials.Look],
    regions: dict[str, IntArray],
) -> None:
    """Write the hero file; `times` None writes the rest pose without animation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{path.name}")
    stage = new_stage(temporary)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    if times is not None:
        stage.SetStartTimeCode(float(np.floor(times.min())))
        stage.SetEndTimeCode(float(np.ceil(times.max())))
    bound = {name: materials.author(stage, f"{LOOKS}/{name}", look) for name, look in sorted(looks.items())}
    for part in parts:
        mesh = _mesh(stage, part)
        _points(mesh, samples[part.name], times)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound[part.look])
        if part.name == "body":
            _regions(mesh, regions)
    stage.GetRootLayer().Save()
    del stage
    temporary.replace(path)
