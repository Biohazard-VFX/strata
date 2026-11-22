"""Deformation motion blur: every prim whose points move gets Karma geometry samples.

Karma renders deformation blur only for prims whose inherited `karma:object:geosamples` exceeds 1;
transforms are sampled by `karma:object:xformsamples` in the render settings. Applied once to the
composed shot, so every hero, groom, hairstyle and future deforming generator is covered alike.
"""

from pxr import Sdf, Usd, UsdGeom

GEO_SAMPLES = 3
PRIMVAR = "karma:object:geosamples"


def geometry_samples(prim: Usd.Prim) -> int:
    """The geometry sample count Karma sees on `prim`: its own or an inherited constant primvar."""
    primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(PRIMVAR)
    return int(primvar.Get()) if primvar and primvar.HasValue() else 1


def deforming(stage: Usd.Stage) -> list[Usd.Prim]:
    """Point-based prims whose points vary over time, through references, sublayers and clips."""
    return [
        prim
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if UsdGeom.PointBased(prim) and UsdGeom.PointBased(prim).GetPointsAttr().ValueMightBeTimeVarying()
    ]


def apply(stage: Usd.Stage) -> None:
    """Author geometry samples on every deforming prim that lacks them, in the stage's edit target."""
    for prim in deforming(stage):
        if geometry_samples(prim) >= GEO_SAMPLES:
            continue
        if prim.IsInstanceProxy():
            raise ValueError(f"{prim.GetPath()}: deforming geometry inside an instance cannot be sampled")
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar(PRIMVAR, Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant).Set(
            GEO_SAMPLES
        )
