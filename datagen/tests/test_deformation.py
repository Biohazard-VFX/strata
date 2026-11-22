import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.library.catalogue import Library
from strata_datagen.scene import compose
from strata_datagen.scene.spec import ShotSpec
from strata_datagen.usdutil import ints, new_stage, vec3f
from tests.test_clip_heroes import clip_library
from tests.test_determinism import RENDER, library
from tests.test_hairstyle import hair_library

PRIMVAR = "karma:object:geosamples"


def write_animated_hero(path: Path, keys: int) -> None:
    """A pose whose skin triangle and one groom curve move at every time code 0..keys-1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = new_stage(path)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/hero").GetPrim())
    skin = UsdGeom.Mesh.Define(stage, "/hero/skin")
    skin.CreateFaceVertexCountsAttr(ints(np.array([3])))
    skin.CreateFaceVertexIndicesAttr(ints(np.arange(3)))
    groom = UsdGeom.BasisCurves.Define(stage, "/hero/groom")
    groom.CreateCurveVertexCountsAttr(ints(np.array([2])))
    for time in range(keys):
        skin.GetPointsAttr().Set(vec3f(np.eye(3) * (1.0 + 0.1 * time)), Usd.TimeCode(time))
        groom.GetPointsAttr().Set(
            vec3f(np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.01 * time]])), Usd.TimeCode(time)
        )
    stage.GetRootLayer().Save()


def composed(lib: Library, root: Path, spec: ShotSpec) -> Usd.Stage:
    shot = compose.plan(spec, lib, root)
    return Usd.Stage.Open(str(compose.write(shot, lib, root, RENDER)))


def geometry_samples(prim: Usd.Prim) -> int:
    primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(PRIMVAR)
    return int(primvar.Get()) if primvar and primvar.HasValue() else 1


def deforming(stage: Usd.Stage) -> dict[str, int]:
    """Every prim whose points move in the composed shot, with the geometry samples Karma sees."""
    return {
        str(prim.GetPath()): geometry_samples(prim)
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if UsdGeom.PointBased(prim) and UsdGeom.PointBased(prim).GetPointsAttr().ValueMightBeTimeVarying()
    }


class DeformationBlurTest(unittest.TestCase):
    def assert_sampled(self, stage: Usd.Stage, expected: set[str]) -> None:
        found = deforming(stage)
        self.assertTrue(expected <= set(found), sorted(found))
        self.assertEqual({p: n for p, n in found.items() if n < 3}, {})
        self.assertEqual(geometry_samples(stage.GetPrimAtPath("/World/terrain")), 1)

    def test_animated_creature_skin_and_groom_get_geometry_samples(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = library()
            for pose in lib.heroes[0].poses:
                write_animated_hero(root / pose.file, 6)
            stage = composed(lib, root, ShotSpec(seed=3, frame_count=24))
            self.assert_sampled(stage, {"/World/hero/skin", "/World/hero/groom"})

    def test_animated_human_clip_gets_geometry_samples(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = clip_library()
            for pose in lib.heroes[0].poses:
                write_animated_hero(root / pose.file, 42)
            stage = composed(lib, root, ShotSpec(seed=11, pose="walk_x", frame_count=24))
            self.assert_sampled(stage, {"/World/hero/skin", "/World/hero/groom"})

    def test_morphing_hairstyle_on_a_static_hero_gets_geometry_samples(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = hair_library(root)
            stage = composed(lib, root, ShotSpec(seed=11))
            hair = {p for p in deforming(stage) if p.startswith("/World/hero/hair/")}
            self.assertTrue(hair)
            self.assert_sampled(stage, hair)


if __name__ == "__main__":
    unittest.main()
