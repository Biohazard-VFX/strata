import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from strata_datagen.library import curves, morph, motion_chain


class CurvesTest(unittest.TestCase):
    def test_mixture_is_convex_smooth_and_seeded(self) -> None:
        seconds = np.arange(0.0, 30.0, 1.0 / 72.0)
        w = curves.mixture(np.random.default_rng(4), seconds, 4, (8.0, 30.0), 2.5)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)
        self.assertTrue(np.all(w > 0.0))
        self.assertLess(np.abs(np.diff(w, axis=0)).max(), 0.01)
        self.assertGreater(np.ptp(w[:, 0]), 0.2)
        np.testing.assert_array_equal(
            w, curves.mixture(np.random.default_rng(4), seconds, 4, (8.0, 30.0), 2.5)
        )

    def test_mix_chain_reproduces_weights(self) -> None:
        w = curves.mixture(np.random.default_rng(1), np.linspace(0.0, 10.0, 50), 3, (2.0, 5.0), 2.0)
        layers = np.array([1.0, 10.0, 100.0])
        m = morph.mix_weights(w)
        colour = np.full(len(w), layers[0])
        for i in range(m.shape[1]):
            colour = colour * (1.0 - m[:, i]) + layers[i + 1] * m[:, i]
        np.testing.assert_allclose(colour, w @ layers)


class BlendTest(unittest.TestCase):
    def test_crossfade_between_distant_poses_is_continuous(self) -> None:
        frames = 72
        a = Rotation.from_rotvec(np.tile([0.0, 0.0, 0.1], (frames, 1))).as_matrix()[:, None]
        spin = np.linspace(2.9, 3.4, frames)
        b = Rotation.from_rotvec(np.stack([np.zeros(frames), np.zeros(frames), spin], axis=-1)).as_matrix()[
            :, None
        ]
        w = np.linspace(0.0, 1.0, frames)
        out = motion_chain.blend(a, b, w * w * (3.0 - 2.0 * w))
        np.testing.assert_allclose(out[0], a[0], atol=1e-9)
        np.testing.assert_allclose(out[-1], b[-1], atol=1e-9)
        steps = np.linalg.norm(np.diff(out.reshape(frames, -1), axis=0), axis=1)
        self.assertLess(steps.max(), 0.15)


if __name__ == "__main__":
    unittest.main()
