import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strata_contract.layers import reduce_samples
from strata_datagen.exr import read_flat
from strata_datagen.render import husk, pack
from tests.test_units import write_deep


class LayerRuleTest(unittest.TestCase):
    def test_close_samples_merge_and_a_large_depth_gap_splits(self) -> None:
        depth = np.array([3.0, 1.0, 1.05])
        alpha = np.array([1.0, 0.5, 0.5])
        rgb = np.array([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        layers = reduce_samples(depth, alpha, rgb, np.array([3]), (1, 1))
        self.assertAlmostEqual(float(layers.alpha[0, 0, 0]), 0.75, places=6)
        self.assertAlmostEqual(
            float(layers.inverse_depth[0, 0, 0]), (0.5 / 1.0 + 0.25 / 1.05) / 0.75, places=5
        )
        self.assertAlmostEqual(float(layers.rgb[0, 0, 0, 0]), 0.5 + 0.5 * 0.1, places=6)
        self.assertAlmostEqual(float(layers.alpha[1, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(layers.inverse_depth[1, 0, 0]), 1.0 / 3.0, places=6)
        self.assertEqual(float(layers.alpha[2:].max()), 0.0)


class PackTest(unittest.TestCase):
    def test_every_frame_gets_layers_and_deep_is_kept_on_the_stride(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp)
            (shot / "deep").mkdir()
            write_deep(shot / "deep" / "0001.exr", [[(0.5, 1.0, 1000)], [(0.2, 0.5, 2)]])
            for frame in (2, 3):
                shutil.copy(shot / "deep" / "0001.exr", shot / "deep" / f"{frame:04d}.exr")
            kept = pack.pack(shot, range(1, 4), 1, 2)
            self.assertEqual(kept, [1, 3])
            self.assertEqual(sorted(p.name for p in (shot / "deep").iterdir()), ["0001.exr", "0003.exr"])
            layers = read_flat(shot / "layers" / "0002.exr")
            np.testing.assert_allclose(layers["L0.A"][0], [1.0, 0.5])
            np.testing.assert_allclose(layers["L0.invdepth"][0], [0.5, 0.5])


class ChunkTest(unittest.TestCase):
    def test_a_long_shot_is_split_into_farm_tasks_covering_every_frame(self) -> None:
        root = Path("/build")
        tasks = husk.jobs(root / "shots" / "s", root, tuple(range(1, 721)), 60, 24)
        self.assertEqual(len(tasks), 12)
        covered = [f for t in tasks for f in range(int(t.command[4]), int(t.command[4]) + int(t.command[6]))]
        self.assertEqual(covered, list(range(1, 721)))
        self.assertEqual(tasks[-1].post[-4:], ["--shot-first", "1", "--stride", "24"])


if __name__ == "__main__":
    unittest.main()
