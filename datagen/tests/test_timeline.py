import unittest

import numpy as np

from strata_datagen import timeline
from strata_datagen.fields import Ground
from strata_datagen.generators import camera
from strata_datagen.qa import checks
from strata_datagen.rng import stream


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


if __name__ == "__main__":
    unittest.main()
