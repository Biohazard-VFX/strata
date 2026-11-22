import tempfile
import unittest
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

from strata_datagen import regions
from strata_datagen.generators import materials
from strata_datagen.library import human_regions, human_usd, humans, mocap, retarget

ASF = """:version 1.10
:units
  mass 1.0
  length 0.5
  angle deg
:root
   order TX TY TZ RX RY RZ
   axis XYZ
:bonedata
  begin
     id 1
     name spine
     direction 0 1 0
     length 10
     axis 0 0 0 XYZ
    dof rx ry rz
  end
  begin
     id 2
     name arm
     direction 1 0 0
     length 5
     axis 0 0 90 XYZ
    dof rx
  end
:hierarchy
  begin
    root spine
    spine arm
  end
"""

AMC = """:FULLY-SPECIFIED
:DEGREES
1
root 0 20 0 0 0 0
spine 0 0 0
arm 0
2
root 0 20 0 0 90 0
spine 0 0 0
arm 90
"""


def _skeleton(folder: Path) -> tuple[mocap.Skeleton, mocap.Motion]:
    (folder / "s.asf").write_text(ASF)
    (folder / "s.amc").write_text(AMC)
    return mocap.load(folder / "s.asf", folder / "s.amc")


class MocapTest(unittest.TestCase):
    def test_units_hierarchy_and_rotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skeleton, motion = _skeleton(Path(tmp))
        inch = 0.0254
        self.assertEqual(skeleton.order, ("root", "spine", "arm"))
        np.testing.assert_allclose(motion.heads[0, 0], [0.0, 40 * inch, 0.0])
        np.testing.assert_allclose(motion.tails[0, 1], [0.0, 60 * inch, 0.0], atol=1e-9)
        np.testing.assert_allclose(motion.tails[0, 2], [10 * inch, 60 * inch, 0.0], atol=1e-9)
        # Frame 2: root yaw +90 maps +X to -Z; the rolled arm rx adds 90 degrees more yaw: +X ends at -X.
        np.testing.assert_allclose(motion.tails[1, 2] - motion.heads[1, 2], [-10 * inch, 0.0, 0.0], atol=1e-9)


class RetargetTest(unittest.TestCase):
    def _rig(self) -> retarget.Rig:
        names = tuple(retarget.CMU_MB)
        heads = np.zeros((len(names), 3))
        tails = np.zeros((len(names), 3))
        tails[:, 1] = 0.1
        parents = np.full(len(names), 0, dtype=np.int64)
        parents[0] = -1
        return retarget.Rig(names, parents, heads, tails)

    def test_swing_maps_direction(self) -> None:
        a, b = np.array([0.3, -1.0, 0.2]), np.array([1.0, 0.1, 0.0])
        moved = retarget.swing(a, b) @ (a / np.linalg.norm(a))
        np.testing.assert_allclose(moved, b / np.linalg.norm(b), atol=1e-12)

    def test_skin_follows_single_bone(self) -> None:
        rig = self._rig()
        turn = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        rotations = np.repeat(np.eye(3)[None, None], len(rig.names), axis=1)
        rotations[0, 3] = turn
        heads = np.zeros((1, len(rig.names), 3))
        heads[0, 3] = [0.0, 0.0, 2.0]
        pose = retarget.Pose(rotations, heads)
        rest = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        joints = np.array([[3, 0], [0, 3]])
        weights = np.array([[1.0, 0.0], [0.5, 0.5]])
        out = retarget.skin(rest, joints, weights, rig.heads, pose)[0]
        np.testing.assert_allclose(out[0], turn @ rest[0] + [0.0, 0.0, 2.0])
        np.testing.assert_allclose(out[1], 0.5 * rest[1] + 0.5 * (turn @ rest[1] + [0.0, 0.0, 2.0]))

    def test_window_keeps_shutter_samples_inside_the_take(self) -> None:
        motion = mocap.Motion(
            ("root",), np.zeros((500, 1, 3, 3)), np.zeros((500, 1, 3)), np.zeros((500, 1, 3))
        )
        start, length = humans.clip_window(motion, humans.MotionOption("x", 0), 96)
        times = start + ((length - 1) + humans.SUBFRAMES[-1]) * mocap.FPS / humans.FPS
        first = start + humans.SUBFRAMES[0] * mocap.FPS / humans.FPS
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(times, 499.0)
        self.assertEqual(length, 98)
        _, short = humans.clip_window(motion, humans.MotionOption("x", 400), 96)
        self.assertLess(short, 98)


class RegionsTest(unittest.TestCase):
    def test_face_subset_majority(self) -> None:
        mask = np.array([True, True, False, False, True])
        counts = np.array([4, 3])
        indices = np.array([0, 1, 2, 3, 2, 3, 4])
        np.testing.assert_array_equal(human_regions.face_subset(mask, counts, indices), [0])

    def test_scalp_face_body_partition(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.uniform([-0.1, 1.45, -0.12], [0.1, 1.8, 0.12], (4000, 3))
        head = (points[:, 1] > 1.5).astype(np.float64)
        neck = 1.0 - head
        groups = {
            "scalp": np.zeros(4000, bool),
            "ears": (np.abs(points[:, 0]) > 0.09) & (np.abs(points[:, 2]) < 0.02),
            "lips": np.zeros(4000, bool),
        }
        marks = human_regions.Landmarks(np.array([0.03, 1.65, 0.09]), np.array([-0.03, 1.65, 0.09]))
        found = human_regions.regions(points, head, neck, groups, marks)
        total = found["scalp"].astype(int) + found["face"] + found["ears"] + found["body"]
        np.testing.assert_array_equal(total[head > 0.5], 1)
        self.assertFalse(np.any(found["brows"] & ~found["face"]))
        span = points[head > 0.5, 1].max() - 1.65
        self.assertTrue(np.all(points[found["scalp"], 1] > 1.65 - human_regions.NAPE_BELOW_EYES * span))


class SamplingTest(unittest.TestCase):
    def _recipe(self) -> humans.Recipe:
        skins = tuple(
            humans.Skin(f"{a}_{s}_{age}", s, (age,), a)
            for a in humans.ANCESTRIES
            for s in humans.SEXES
            for age in humans.AGES
        )
        pools = {"full": ("f",), "tops": ("t",), "bottoms": ("b",), "shoes": ("s",)}
        motions = {
            c: (humans.MotionOption(f"{c}_1", 0), humans.MotionOption(f"{c}_2", 0)) for c in humans.CATEGORIES
        }
        return humans.Recipe(
            seed=3,
            count=24,
            key_prefix="h_",
            blender="blender",
            blender_user_resources=Path(),
            mpfb_data=Path(),
            mpfb_targets=Path(),
            cmu=Path(),
            skins=skins,
            outfits={"male": pools, "female": pools},
            eyes=("brown",),
            clip_frames=96,
            motions=motions,
            licenses=(),
        )

    def _catalog(self) -> humans.Catalog:
        sided: dict[str, object] = {
            "has_left_and_right": True,
            "targets": ["unused"],
            "opposites": {
                f"{s}-{side}": f"{s}-{side}-t" for s in ("positive", "negative") for side in ("left", "right")
            }
            | {"positive-unsided": "", "negative-unsided": ""},
        }
        single: dict[str, object] = {"has_left_and_right": False, "targets": ["shape"]}
        return {g: {"categories": [sided, single] * 2} for g in humans.FACE_GROUPS}

    def test_stratified_and_deterministic(self) -> None:
        recipe, catalog = self._recipe(), self._catalog()
        specs = [humans.sample(recipe, i, catalog) for i in range(24)]
        cells = {(s.skin.ancestry, s.skin.sex, s.skin.ages[0]) for s in specs}
        self.assertEqual(len(cells), 24)
        self.assertEqual(humans.sample(recipe, 7, catalog), specs[7])
        for spec in specs:
            names = [t for t, _ in spec.targets]
            left = [n for n in names if n.endswith("-left-t")]
            self.assertEqual(len(left), len([n for n in names if n.endswith("-right-t")]))
            self.assertEqual(set(spec.motions), set(humans.CATEGORIES))


class HumanUsdTest(unittest.TestCase):
    def test_animated_parts_regions_and_rest(self) -> None:
        part = human_usd.MeshPart(
            "body",
            np.array([4]),
            np.array([0, 1, 2, 3]),
            np.zeros((4, 2)),
            np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            "body",
        )
        times = np.array([-0.25, 0.0, 0.25, 0.75, 1.0])
        samples = {"body": part.rest[None] + times[:, None, None] * np.array([1.0, 0.0, 0.0])}
        look = materials.Look(materials.Textures("albedo.jpg"), subsurface=0.3, subsurface_scale_m=0.01)
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(tmp) / "clip.usdc"
        human_usd.write(
            path,
            parts=[part],
            samples=samples,
            times=times,
            looks={"body": look},
            regions={"scalp": np.array([1.0, 0.5, 0.5, 0.0])},
        )
        stage = Usd.Stage.Open(str(path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/hero/body"))
        self.assertEqual(mesh.GetPointsAttr().GetTimeSamples(), list(times))
        np.testing.assert_allclose(np.asarray(mesh.GetPointsAttr().Get(0.25))[:, 0], part.rest[:, 0] + 0.25)
        rest = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(human_usd.REST_PRIMVAR).Get()
        np.testing.assert_allclose(np.asarray(rest), part.rest)
        self.assertEqual(regions.names(mesh), ("scalp",))
        np.testing.assert_allclose(regions.read(mesh, ("scalp",))["scalp"], [1.0, 0.5, 0.5, 0.0])
        self.assertFalse(UsdGeom.Subset.GetAllGeomSubsets(mesh))
        bound = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0]
        surface = bound.GetPrim().GetChild("surface")
        self.assertAlmostEqual(surface.GetAttribute("inputs:subsurface").Get(), 0.3, places=6)
        self.assertTrue(surface.GetAttribute("inputs:subsurface_color").HasAuthoredConnections())


if __name__ == "__main__":
    unittest.main()
