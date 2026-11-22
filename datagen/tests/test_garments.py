import unittest

import numpy as np

from strata_datagen.library import garments


def _cylinder_body(radius: float, height: float) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    levels = np.arange(0.0, height, 0.005)
    ring = np.stack([radius * np.cos(angle), np.zeros_like(angle), radius * np.sin(angle)], axis=-1)
    return np.concatenate([ring + np.array([0.0, y, 0.0]) for y in levels])


class GarmentTest(unittest.TestCase):
    def test_girth_of_a_cylinder(self) -> None:
        body = _cylinder_body(0.15, 1.8)
        self.assertAlmostEqual(garments.girth(body, 1.0), 2.0 * np.pi * 0.15, delta=0.01)

    def test_measurements_scale_with_height_and_use_measured_girths(self) -> None:
        body = _cylinder_body(0.15, 1.8)
        heads = {
            "Neck": np.array([0.0, 1.5, 0.0]),
            "LeftUpLeg": np.array([0.1, 0.9, 0.0]),
            "LeftShoulder": np.array([0.05, 1.45, 0.0]),
            "LeftArm": np.array([0.2, 1.4, 0.0]),
            "LeftForeArm": np.array([0.4, 1.2, 0.0]),
        }
        template = {"height": 90.0, "arm_length": 30.0, "waist": 1.0, "hips": 1.0, "bust": 1.0}
        out = garments.measure(body, heads, template)
        self.assertAlmostEqual(out["height"], 179.5, delta=0.6)
        self.assertAlmostEqual(out["arm_length"], 30.0 * out["height"] / 90.0, places=3)
        for name in ("waist", "hips", "bust"):
            self.assertAlmostEqual(out[name], 2.0 * np.pi * 15.0, delta=1.0)
        self.assertAlmostEqual(out["arm_pose_angle"], 45.0, places=3)

    def test_sim_grid_hits_shutter_samples_and_resample_is_linear(self) -> None:
        times = (np.arange(5, dtype=np.float64)[:, None] + np.array([-0.25, 0.0, 0.25])[None]).ravel()
        grid = garments.sim_grid(times)
        self.assertTrue(np.all(np.isin(times[::3], grid)))
        values = np.stack([np.full((2, 3), t) for t in grid])
        np.testing.assert_allclose(garments.resample(grid, values, times)[:, 0, 0], times)

    def test_preroll_starts_at_rest_over_the_clip_and_ends_on_it(self) -> None:
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        first = rest + np.array([2.0, 0.5, 3.0])
        roll = garments.preroll(rest, first)
        np.testing.assert_allclose(roll[0], rest + np.array([2.0, 0.0, 3.0]))
        np.testing.assert_allclose(roll[-1], first)
        self.assertEqual(len(roll), garments.SETTLE // 2 + garments.PREROLL + garments.SETTLE)


if __name__ == "__main__":
    unittest.main()
