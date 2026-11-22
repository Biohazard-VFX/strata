import unittest

import numpy as np

from parity import discontinuities


class DiscontinuityMask(unittest.TestCase):
    def test_step_masks_two_pixels_each_side(self):
        reference = np.full((6, 12), .5, np.float32)
        reference[:, 6:] = 1.5
        near = discontinuities(reference, .3, 2)
        np.testing.assert_array_equal(np.flatnonzero(near[0]), [3, 4, 5, 6, 7, 8])
        self.assertTrue((near == near[0]).all())

    def test_step_below_threshold_is_surface(self):
        reference = np.full((6, 12), 1, np.float32)
        reference[:, 6:] = 1.25
        self.assertFalse(discontinuities(reference, .3, 2).any())

    def test_threshold_is_on_log_inverse_depth(self):
        near = np.full((4, 8), 1e-3, np.float32)
        near[:, 4:] = 1.5e-3
        far = near * 1000
        np.testing.assert_array_equal(discontinuities(near, .3, 1), discontinuities(far, .3, 1))
        self.assertTrue(discontinuities(near, .3, 1).any())

    def test_smooth_ramp_is_not_a_discontinuity(self):
        reference = np.exp(np.linspace(0, 5, 40, dtype=np.float32))[None].repeat(5, 0)
        self.assertFalse(discontinuities(reference, .3, 2).any())

    def test_vertical_step_and_zero_boundary(self):
        reference = np.ones((10, 5), np.float32)
        reference[5:] = 3
        np.testing.assert_array_equal(np.flatnonzero(discontinuities(reference, .3, 2)[:, 0]), [2, 3, 4, 5, 6, 7])
        reference[5:] = 0
        np.testing.assert_array_equal(np.flatnonzero(discontinuities(reference, .3, 2)[:, 0]), [2, 3, 4, 5, 6, 7])

    def test_isolated_spike_masks_square(self):
        reference = np.ones((9, 9), np.float32)
        reference[4, 4] = 4
        near = discontinuities(reference, .3, 2)
        expected = np.zeros_like(near)
        expected[1:8, 1:8] = True
        expected[1, 1] = expected[1, 7] = expected[7, 1] = expected[7, 7] = False
        np.testing.assert_array_equal(near, expected)


if __name__ == '__main__':
    unittest.main()
