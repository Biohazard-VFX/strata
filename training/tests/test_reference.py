"""The oracle's sRGB encoding must match the contract bit for bit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import OpenEXR
from strata_train.reference import encode, encode_acescg, load


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

    def test_load_converts_linear_pixels_before_encoding(self) -> None:
        rgb = np.array([[[0.25, 0.5, 2], [-0.25, 0.75, 4]]], np.float32)
        matrix = np.array([[0.6, 0.35, 0.05], [0.07, 0.92, 0.01], [0.02, 0.11, 0.87]], np.float64)
        expected = np.array([[[0.425, 0.4975, 1.8], [0.3125, 0.7125, 3.5575]]], np.float32)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "plate.exr"
            channels = {c: rgb[:, :, i].copy() for i, c in enumerate("RGB")}
            cast("Any", OpenEXR).File({}, channels).write(str(path))
            np.testing.assert_array_equal(load(path, matrix, 0.1), encode_acescg(expected, 0.1))

    def test_nonfinite_samples_clamp(self) -> None:
        values = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
        np.testing.assert_array_equal(encode(values), np.array([1, 0, 0], dtype=np.float32))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with np.load(sys.argv[1]) as fixture:
            for stops, expected in zip(fixture["exposures"], fixture["encoded"], strict=True):
                np.testing.assert_array_equal(
                    encode_acescg(fixture["linear"].transpose(1, 2, 0), float(stops)).transpose(
                        2, 0, 1
                    ),
                    expected,
                )
        print("Reference/training encoder parity: bit-identical")
    else:
        unittest.main()
