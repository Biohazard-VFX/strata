"""The oracle's sRGB encoding must match the contract bit for bit."""

from __future__ import annotations

import sys
import unittest

import numpy as np
from strata_train.reference import encode


class ReferenceTests(unittest.TestCase):
    def test_fixed_fable_values(self) -> None:
        samples = np.array([0, 0.001, 0.18, 1, 1.4, 10, 64, 128, 1000, -1], np.float32)
        expected = np.array(
            [
                0,
                0.00930240005,
                0.395302713,
                0.850128949,
                0.907322526,
                0.989884555,
                0.998465478,
                0.999234796,
                0.999902308,
                0,
            ],
            np.float32,
        )
        np.testing.assert_allclose(encode(samples), expected, rtol=0, atol=6e-8)

    def test_nonfinite_samples_clamp(self) -> None:
        values = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
        np.testing.assert_array_equal(encode(values), np.array([1, 0, 0], dtype=np.float32))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with np.load(sys.argv[1]) as fixture:
            for stops, expected in zip(fixture["exposures"], fixture["encoded"], strict=True):
                np.testing.assert_array_equal(encode(fixture["linear"], float(stops)), expected)
        print("Reference/training encoder parity: bit-identical")
    else:
        unittest.main()
