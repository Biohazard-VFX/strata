import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.generators import morph
from strata_datagen.library import creatures
from strata_datagen.rng import stream
from strata_datagen.scene import creature
from strata_datagen.usdutil import floats, ints, new_stage, vec3f


class SignalTest(unittest.TestCase):
    def test_widths_taper_length_and_gate_density_without_dropping_strands(self) -> None:
        base = np.full((1000, 9), 1e-4)
        rank = np.linspace(0.0, 1.0, 1000, endpoint=False)
        shown = morph.strand_widths(base, rank, 0.5, 0.3, softness=1e-6)
        self.assertEqual(shown.shape, base.shape)
        visible = shown.max(axis=1) > 0
        self.assertAlmostEqual(float(visible.mean()), 0.3, delta=0.01)
        self.assertTrue((shown[visible][:, :4] > 0).all())
        self.assertTrue((shown[:, 5:] == 0).all())


class LoopTest(unittest.TestCase):
    def test_close_loop_finds_the_period_and_matches_ends(self) -> None:
        t = np.arange(40)[:, None, None]
        clip = np.sin(2 * np.pi * t / 13.0) * np.ones((1, 5, 3)) + 0.002 * t
        period = creatures.loop_period(clip, 8)
        self.assertEqual(period, 13)
        closed = creatures.close_loop(clip, period)
        self.assertEqual(len(closed), 14)
        np.testing.assert_allclose(closed[-1], closed[0])

    def test_value_clip_loop_replays_the_cycle_over_a_long_shot(self) -> None:
        with TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "cycle.usdc"
            clip = new_stage(clip_path)
            clip.SetDefaultPrim(UsdGeom.Xform.Define(clip, "/hero").GetPrim())
            curves = UsdGeom.BasisCurves.Define(clip, "/hero/groom")
            curves.CreateCurveVertexCountsAttr(ints(np.array([2])))
            points = curves.CreatePointsAttr(vec3f(np.zeros((2, 3))))
            curves.CreateWidthsAttr(floats(np.full(2, 1e-3)))
            for frame in range(5):
                points.Set(vec3f(np.full((2, 3), float(frame % 4))), float(frame))
            clip.GetRootLayer().Save()
            shot = Usd.Stage.CreateInMemory()
            hero = shot.DefinePrim("/World/hero")
            hero.GetReferences().AddReference(str(clip_path))
            morph.author_loop(hero, str(clip_path), "/hero", period=4, first=0, last=30)
            attribute = UsdGeom.BasisCurves(shot.GetPrimAtPath("/World/hero/groom")).GetPointsAttr()
            values = [float(np.asarray(attribute.Get(float(f)))[0, 0]) for f in range(30)]
            self.assertEqual(values, [float(f % 4) for f in range(30)])
            morph.author_coat(
                shot,
                "/World/hero/groom",
                np.full((1, 2), 1e-3),
                morph=morph.CoatMorph(morph.Range(0.5, 1.0), morph.Range(0.5, 1.0), 8.0, 12),
                frames=(0, 30),
                rng=stream(2, "m"),
            )
            widths = UsdGeom.BasisCurves(shot.GetPrimAtPath("/World/hero/groom")).GetWidthsAttr()
            self.assertEqual(widths.GetTimeSamples(), [0.0, 12.0, 24.0, 30.0])


class CreatureSceneTest(unittest.TestCase):
    def test_animate_loops_the_pose_and_keys_widths_in_a_binary_sublayer(self) -> None:
        with TemporaryDirectory() as tmp:
            asset = Path(tmp) / "pose.usdc"
            pose = new_stage(asset)
            pose.SetDefaultPrim(UsdGeom.Xform.Define(pose, "/hero").GetPrim())
            skin = UsdGeom.Mesh.Define(pose, "/hero/skin").CreatePointsAttr(vec3f(np.zeros((3, 3))))
            for frame in range(7):
                skin.Set(vec3f(np.full((3, 3), float(frame % 6))), float(frame))
            curves = UsdGeom.BasisCurves.Define(pose, "/hero/groom")
            curves.CreateCurveVertexCountsAttr(ints(np.array([4, 4])))
            curves.CreatePointsAttr(vec3f(np.zeros((8, 3))))
            curves.CreateWidthsAttr(floats(np.full(8, 1e-3)))
            ranges = {"length": [0.5, 1.0], "density": [0.6, 1.0], "period_s": 8.0, "key_step": 48}
            curves.GetPrim().SetCustomDataByKey("morph", ranges)
            pose.GetRootLayer().Save()
            shot_path = Path(tmp) / "scene" / "shot.usda"
            shot_path.parent.mkdir()
            shot = new_stage(shot_path)
            shot.DefinePrim("/World/hero").GetReferences().AddReference(str(asset))
            record = creature.animate(
                shot, "/World/hero", asset, asset_path=str(asset), frames=(1, 100), seed=3
            )
            shot.GetRootLayer().Save()
            self.assertEqual(record["loop_frames"], 6)
            self.assertTrue((shot_path.parent / creature.COAT_LAYER).exists())
            self.assertNotIn("widths", shot_path.read_text())
            reopened = Usd.Stage.Open(str(shot_path))
            widths = UsdGeom.BasisCurves(reopened.GetPrimAtPath("/World/hero/groom")).GetWidthsAttr()
            self.assertEqual(widths.GetTimeSamples(), [0.0, 48.0, 96.0, 101.0])
            points = UsdGeom.Mesh(reopened.GetPrimAtPath("/World/hero/skin")).GetPointsAttr()
            self.assertEqual(float(np.asarray(points.Get(14.0))[0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
