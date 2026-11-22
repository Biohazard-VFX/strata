import unittest
from dataclasses import replace

import numpy as np

from strata_datagen.fields import Ground
from strata_datagen.generators import scatter, terrain
from strata_datagen.generators.scatter import Layer, Prototype
from strata_datagen.rng import stream


def _flat_ground(size_m: float = 40.0) -> Ground:
    count = int(size_m / 0.5) + 1
    ones = np.ones((count, count))
    return Ground(np.zeros((count, count)), 0.5, {"flat": ones})


def _layer(*footprints: float) -> Layer:
    prototypes = tuple(Prototype(f"p{i}", "a.usd", "/m", f, f) for i, f in enumerate(footprints))
    return Layer("rocks", prototypes, 2.0, 15.0, (0.0, 0.0), ("flat",), (0.25, 2.5), 1.0, 0.0)


class TerrainTest(unittest.TestCase):
    def test_ground_beyond_the_scattered_area_never_rises_above_the_centre(self) -> None:
        for seed in range(6):
            params = replace(terrain.sample_params(stream(seed, "t")), size_m=200.0, cell_m=1.0)
            ground = terrain.build(params, stream(seed, "n"))
            axis = np.linspace(-100.0, 100.0, ground.heights.shape[0])
            x, z = np.meshgrid(axis, axis)
            radius = np.hypot(x, z)
            centre = float(ground.heights[radius < 2.0].mean())
            self.assertLess(
                float(ground.heights[radius > 0.95 * 100.0].max()), centre - params.edge_drop_m + 0.5
            )
            outer = (radius > 60.0) & (radius < 95.0)
            self.assertLess(float(ground.heights[outer].max()), centre + params.amplitude_m)
            self.assertLessEqual(float(ground.masks["low"].max()), 1.0)


class ScatterScaleTest(unittest.TestCase):
    def test_instances_stay_within_the_native_scale_range(self) -> None:
        instances = scatter.place(_flat_ground(), _layer(0.2, 1.0, 3.4), [], stream(1, "s"))
        self.assertGreater(len(instances), 100)
        scale = instances.scales[:, 0]
        low, high = scatter.NATIVE_SCALE
        self.assertTrue(((scale >= low - 1e-9) & (scale <= high + 1e-9)).all())
        footprint = np.array([0.2, 1.0, 3.4])[instances.prototype_indices] * scale
        self.assertTrue(((footprint >= 0.25 - 1e-9) & (footprint <= 2.5 + 1e-9)).all())
        self.assertLess(float(scale[instances.prototype_indices == 2].max()), 2.5 / 3.4 + 1e-9)

    def test_models_that_would_be_shrunk_to_pillars_are_refused(self) -> None:
        self.assertFalse(scatter.fits(18.0, (0.25, 2.5)))
        self.assertFalse(scatter.fits(0.1, (0.25, 2.5)))
        self.assertTrue(scatter.fits(3.4, (0.25, 2.5)))
        with self.assertRaises(ValueError):
            scatter.place(_flat_ground(), _layer(1.0, 18.0), [], stream(1, "s"))


if __name__ == "__main__":
    unittest.main()
