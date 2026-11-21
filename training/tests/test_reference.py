"""The oracle's sRGB encoding must match the contract bit for bit."""

from __future__ import annotations

import unittest

import numpy as np
from strata_contract import srgb_encode, srgb_samples
from strata_train.reference import encode


class ReferenceTests(unittest.TestCase):
    def test_srgb_matches_contract_bits(self) -> None:
        samples = np.array(srgb_samples(), dtype=np.float32)
        expected = np.array([srgb_encode(float(value)) for value in samples], dtype=np.float32)
        np.testing.assert_array_equal(encode(samples).view(np.uint32), expected.view(np.uint32))

    def test_nonfinite_samples_clamp(self) -> None:
        values = np.array([np.inf, -np.inf, np.nan, 2], dtype=np.float32)
        np.testing.assert_array_equal(encode(values), np.array([1, 0, 0, 1], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
