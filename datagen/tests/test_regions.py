import unittest

import numpy as np
from pxr import Sdf, Usd, UsdGeom

from strata_datagen import regions
from strata_datagen.usdutil import floats, ints, vec3f

COUNTS = np.array([4, 4, 4])
INDICES = np.array([0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6])


def strip(stage: Usd.Stage) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, "/skin")
    points = np.array([[x, y, 0.0] for y in (0.0, 1.0) for x in range(4)], dtype=np.float64)
    mesh.CreatePointsAttr(vec3f(points))
    mesh.CreateFaceVertexCountsAttr(ints(COUNTS))
    mesh.CreateFaceVertexIndicesAttr(ints(INDICES))
    return mesh


class RegionsTest(unittest.TestCase):
    def test_face_regions_become_soft_vertex_weights_and_back(self) -> None:
        faces = np.array([True, False, False])
        weight = regions.from_faces(faces, COUNTS, INDICES, 8)
        np.testing.assert_allclose(weight, [1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0])
        np.testing.assert_array_equal(regions.face_mask(weight, COUNTS, INDICES), faces)
        middle = regions.from_faces(np.array([False, True, False]), COUNTS, INDICES, 8)
        np.testing.assert_array_equal(regions.face_mask(middle, COUNTS, INDICES), [False, True, False])

    def test_authored_regions_read_back_and_bad_ones_fail(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        mesh = strip(stage)
        weight = np.linspace(0.0, 1.0, 8)
        regions.author(mesh, {"scalp": weight, "beard": 1.0 - weight})
        self.assertEqual(regions.names(mesh), ("beard", "scalp"))
        np.testing.assert_allclose(regions.read(mesh, ("scalp",))["scalp"], weight, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "no primvar region_brows"):
            regions.read(mesh, ("brows",))
        api = UsdGeom.PrimvarsAPI(mesh)
        api.CreatePrimvar("region_lips", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.uniform).Set(
            floats(np.zeros(3))
        )
        with self.assertRaisesRegex(ValueError, "uniform interpolation"):
            regions.read(mesh, ("lips",))
        api.CreatePrimvar("region_ears", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
            floats(np.full(8, 1.5))
        )
        with self.assertRaisesRegex(ValueError, "leaves"):
            regions.read(mesh, ("ears",))
        api.CreatePrimvar("region_face", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(
            ints(np.zeros(8, dtype=np.int64))
        )
        with self.assertRaisesRegex(ValueError, "not float"):
            regions.read(mesh, ("face",))
        with self.assertRaisesRegex(ValueError, "lower-case identifier"):
            regions.author(mesh, {"Scalp": weight})


if __name__ == "__main__":
    unittest.main()
