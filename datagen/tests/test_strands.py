import json
import unittest
from pathlib import Path

import numpy as np

from strata_datagen import strands
from strata_datagen.usdutil import FloatArray

REFERENCE = json.loads((Path(__file__).parent / "data" / "blender_hair_ops.json").read_text())
CURVES = np.asarray(REFERENCE["input"], dtype=np.float64)
GUIDE = np.zeros(len(CURVES), dtype=np.int64)
TOLERANCE_M = 2e-5


def _expected(case: str) -> np.ndarray:
    return np.asarray(REFERENCE["cases"][case]["output"], dtype=np.float64)


def _dot(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.einsum("...d,...d->...", a, b)


def _norm(a: FloatArray) -> FloatArray:
    return np.sqrt(_dot(a, a))


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


class ShapeRangeTest(unittest.TestCase):
    def test_landmark_shapes(self) -> None:
        t = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(strands.shape_range(t, 0.0), np.ones_like(t))
        np.testing.assert_allclose(strands.shape_range(t, 0.5), t, atol=1e-6)
        np.testing.assert_allclose(strands.shape_range(t, -0.5), 1.0 - t, atol=1e-6)

    def test_late_shapes_rise_at_the_tip(self) -> None:
        t = np.array([0.5])
        self.assertLess(strands.shape_range(t, 0.9)[0], strands.shape_range(t, 0.5)[0])
        self.assertGreater(strands.shape_range(t, 0.2)[0], strands.shape_range(t, 0.5)[0])


class BlenderParityTest(unittest.TestCase):
    """Outputs of Blender's CC0 hair node groups on a three-curve clump with curve 0 as the guide."""

    def test_clump_matches_blender(self) -> None:
        for case, kwargs in (
            ("clump_late", {"factor": 1.0, "shape": 0.8}),
            ("clump_root", {"factor": 0.8, "shape": -0.3}),
            ("clump_falloff", {"factor": 1.0, "shape": 0.25, "falloff": 0.01, "threshold": 0.004}),
        ):
            with self.subTest(case=case):
                out = strands.clump(CURVES, GUIDE, tip_spread=0.0, rng=_rng(), **kwargs)
                self.assertGreater(np.abs(_expected(case) - CURVES).max(), 10 * TOLERANCE_M)
                np.testing.assert_allclose(out, _expected(case), atol=TOLERANCE_M)

    def test_trim_matches_blender(self) -> None:
        out = strands.trim(CURVES, np.full(len(CURVES), 0.6))
        np.testing.assert_allclose(out, _expected("trim"), atol=TOLERANCE_M)

    def test_curl_matches_blender_radius_and_turn_rate(self) -> None:
        """The helix phase depends on each tool's root normal, so compare radius and winding rate."""
        frequency = REFERENCE["cases"]["curl"]["params"]["Frequency"]
        radius = REFERENCE["cases"]["curl"]["params"]["Radius"]
        turns = strands.CURL_TURNS_PER_FREQUENCY * frequency
        out = strands.curl(CURVES, GUIDE, radius=radius, turns_per_m=turns, rng=_rng())
        for curves in (out, _expected("curl")):
            np.testing.assert_allclose(_norm(curves[0] - CURVES[0]), radius, atol=TOLERANCE_M)
        tangent, normal, binormal = strands.frames(CURVES[:1])
        mine, blender = out[0] - CURVES[0], _expected("curl")[0] - CURVES[0]
        angle = [np.unwrap(np.arctan2(_dot(o, normal[0]), _dot(o, binormal[0]))) for o in (mine, blender)]
        np.testing.assert_allclose(np.diff(angle[0]), np.diff(angle[1]), atol=0.02)
        np.testing.assert_allclose(_dot(mine, tangent[0]), 0.0, atol=1e-9)


class OperatorTest(unittest.TestCase):
    def test_resample_spaces_points_evenly_and_keeps_ends(self) -> None:
        out = strands.resample(CURVES, 11)
        steps = strands.segment_lengths(out)
        np.testing.assert_allclose(steps, steps[:, :1] * np.ones_like(steps), rtol=0.02)
        np.testing.assert_allclose(out[:, [0, -1]], CURVES[:, [0, -1]], atol=1e-12)

    def test_restore_lengths_recovers_segment_lengths(self) -> None:
        bent = strands.frizz(CURVES, distance=0.01, shape=0.0, rng=_rng())
        fixed = strands.restore_lengths(bent, CURVES)
        np.testing.assert_allclose(
            strands.segment_lengths(fixed), strands.segment_lengths(CURVES), atol=1e-12
        )
        self.assertGreater(
            np.abs(strands.segment_lengths(bent) - strands.segment_lengths(CURVES)).max(), 1e-4
        )

    def test_frizz_and_noise_keep_roots_and_grow_towards_tips(self) -> None:
        for out in (
            strands.frizz(CURVES, distance=0.005, shape=0.5, rng=_rng()),
            strands.noise(CURVES, distance=0.005, scale=10.0, shape=0.5, rng=_rng()),
        ):
            moved = np.linalg.norm(out - CURVES, axis=-1)
            np.testing.assert_allclose(moved[:, 0], 0.0, atol=1e-12)
            self.assertGreater(moved[:, -1].mean(), moved[:, 1].mean())

    def test_braid_strands_cross_each_other(self) -> None:
        axis = np.stack([np.zeros(40), -np.linspace(0.0, 0.3, 40), np.zeros(40)], axis=-1)
        out = strands.braid(
            np.zeros((6, 40, 3)), axis, radius=0.01, turns_per_m=10.0, strand_radius=0.0, rng=_rng()
        )
        x = out[:3, :, 0]
        order = np.argsort(x, axis=0)
        self.assertGreater(len({tuple(column) for column in order.T}), 3)
        np.testing.assert_allclose(out[0], out[3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
