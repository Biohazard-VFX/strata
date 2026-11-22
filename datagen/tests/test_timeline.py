import unittest

import numpy as np

from strata_datagen import timeline
from strata_datagen.fields import Ground
from strata_datagen.generators import camera
from strata_datagen.qa import checks
from strata_datagen.rng import stream
from strata_datagen.scene import compose


class CurveTest(unittest.TestCase):
    def test_curves_stay_in_range_and_move_by_bounded_steps(self) -> None:
        t = timeline.sample_times(722)
        for seed in range(20):
            curve = timeline.curve(stream(seed, "c"), 31.0, 6.0, -60.0, 60.0)
            values = curve(t)
            self.assertTrue(((values >= -60.0) & (values <= 60.0)).all())
            self.assertLessEqual(timeline.max_step(values), curve.max_slope / timeline.FPS + 1e-9)
            self.assertGreater(np.ptp(values), 10.0)
            second = np.abs(np.diff(values, 2)).max()
            self.assertLess(second, 0.05)

    def test_max_step_finds_a_jump(self) -> None:
        self.assertEqual(timeline.max_step(np.array([0.0, 0.1, 2.1, 2.0])), 2.0)


class CameraPathTest(unittest.TestCase):
    def test_a_long_path_follows_the_target_without_jumps(self) -> None:
        ground = Ground(np.zeros((401, 401)), 0.5, {})
        frames = tuple(range(1, 721))
        targets = np.stack([np.linspace(0.0, 15.0, 722), np.full(722, 0.4), np.zeros(722)], axis=-1)
        for seed in range(4):
            choices = camera.CameraChoices(depth_of_field=True)
            track = camera.sample(stream(seed, "c"), ground, targets, 1.6, frames=frames, choices=choices)
            self.assertEqual(track.move, camera.PATH)
            records = [
                {
                    "world_to_camera": w2c.tolist(),
                    "focal_length_mm": lens[0],
                    "fstop": lens[1],
                    "focus_distance_m": lens[2],
                }
                for w2c, lens in zip(track.world_to_camera(), track.frame_lenses(), strict=True)
            ]
            steps = checks.camera_steps(records)
            self.assertTrue(all(steps[k] <= checks.STEP_LIMITS[k] for k in steps), steps)
            self.assertGreater(float(np.ptp(track.frame_lenses()[:, 1])), 1.0)
            records[300]["fstop"] = 40.0
            self.assertGreater(checks.camera_steps(records)["fstop"], checks.STEP_LIMITS["fstop"])


class FramingTest(unittest.TestCase):
    def test_camera_distance_scales_with_the_subject(self) -> None:
        ground = Ground(np.zeros((41, 41)), 0.5, {})
        target = np.array([0.0, 0.01, 0.0])
        for move in ("orbit", camera.PATH):
            frames = tuple(range(1, 121))
            choices = camera.CameraChoices(move=move)
            small = camera.sample(stream(3, "c"), ground, target, 0.03, frames=frames, choices=choices)
            large = camera.sample(stream(3, "c"), ground, target, 1.6, frames=frames, choices=choices)
            ratio = small.frame_lenses()[:, 2] / large.frame_lenses()[:, 2]
            np.testing.assert_allclose(ratio, 0.03 / 1.6, rtol=0.05)
            np.testing.assert_allclose(small.frame_lenses()[:, 0], large.frame_lenses()[:, 0], rtol=1e-6)


class SamplesTest(unittest.TestCase):
    def test_depth_of_field_or_motion_blur_doubles_the_samples(self) -> None:
        ground = Ground(np.zeros((41, 41)), 0.5, {})
        target = np.array([0.0, 0.4, 0.0])
        for dof, blur, expected in ((False, False, 64), (True, False, 128), (False, True, 128)):
            choices = camera.CameraChoices(move="locked", depth_of_field=dof, motion_blur=blur)
            track = camera.sample(stream(1, "c"), ground, target, 1.6, frames=(1, 2), choices=choices)
            self.assertEqual(compose.samples_for(64, track), expected)


if __name__ == "__main__":
    unittest.main()
