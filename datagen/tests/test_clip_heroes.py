import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Sdf

from strata_datagen.generators import human
from strata_datagen.library.catalogue import Hero, HeroPose, Library
from strata_datagen.scene import compose
from strata_datagen.scene.spec import ShotSpec
from tests.test_determinism import PROVENANCE, RENDER, library

CLIP_FRAMES = 40


def _track(frames: int, speed: float) -> tuple[tuple[float, float, float], ...]:
    z = (np.arange(frames + 2) - (frames + 2) // 2) * speed / 24.0
    return tuple((0.0, 0.95, float(v)) for v in z)


def clip_library() -> Library:
    base = library()
    clips = (
        HeroPose(
            "walk_x",
            "assets/hero/walk.usdc",
            (-0.4, 0.0, -0.3),
            (0.4, 1.8, 0.3),
            0,
            CLIP_FRAMES,
            _track(CLIP_FRAMES, 1.4),
        ),
        HeroPose("run_x", "assets/hero/run.usdc", (-0.4, 0.0, -0.5), (0.4, 1.8, 0.5), 0, 30, _track(30, 3.5)),
    )
    return replace(base, heroes=(Hero("person", "", clips, PROVENANCE),))


class ClipHeroTest(unittest.TestCase):
    def test_clip_reference_offset_ground_and_blur_samples(self) -> None:
        lib = clip_library()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ShotSpec(seed=11, pose="walk_x", frame_count=24)
            shot = compose.plan(spec, lib, root)
            assert shot.placement is not None
            self.assertGreaterEqual(shot.placement.start, 1)
            self.assertLessEqual(shot.placement.start + len(shot.frames), CLIP_FRAMES + 1)
            ground = shot.ground.height(shot.motion.positions[:, 0], shot.motion.positions[:, 2])
            np.testing.assert_allclose(shot.motion.positions[:, 1], ground)
            self.assertAlmostEqual(shot.motion.speed_mps, 1.4, places=6)
            path = compose.write(shot, lib, root, RENDER)
            layer = Sdf.Layer.FindOrOpen(str(path))  # pyright: ignore[reportUnknownMemberType]
            prim = layer.GetPrimAtPath("/World/hero")
            refs = prim.referenceList.prependedItems
            self.assertEqual(refs[0].layerOffset.offset, shot.frames[0] - shot.placement.start)
            self.assertEqual(prim.attributes["primvars:karma:object:geosamples"].default, human.GEO_SAMPLES)
            self.assertEqual(prim.attributes["primvars:instance_id"].default, 2)
            self.assertFalse(layer.GetPrimAtPath("/World/materials/hero_skin"))

    def test_long_shots_only_pick_clips_that_cover_them(self) -> None:
        lib = clip_library()
        with TemporaryDirectory() as tmp:
            for seed in range(12):
                shot = compose.plan(ShotSpec(seed=seed, frame_count=35), lib, Path(tmp))
                self.assertEqual(shot.pose.pose, "walk_x")

    def test_same_seed_same_placement(self) -> None:
        lib = clip_library()
        with TemporaryDirectory() as tmp:
            a = compose.plan(ShotSpec(seed=5, frame_count=24), lib, Path(tmp))
            b = compose.plan(ShotSpec(seed=5, frame_count=24), lib, Path(tmp))
        assert a.placement is not None and b.placement is not None
        self.assertEqual(
            (a.pose.pose, a.placement.start, a.placement.yaw_deg),
            (b.pose.pose, b.placement.start, b.placement.yaw_deg),
        )
        np.testing.assert_array_equal(a.placement.offset, b.placement.offset)


if __name__ == "__main__":
    unittest.main()
