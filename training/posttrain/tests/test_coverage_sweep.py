from itertools import permutations
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from strata_training.eval.coverage import (
    coverage_integrals,
    coverage_regions,
    coverage_samples,
    coverage_summary,
)
from strata_training.eval.selection import validation_improves
from strata_training.types import Layers


def test_opaque_sweep_is_log_distance_and_signed() -> None:
    front = np.array([2.0])[:, None, None]
    alpha = np.ones_like(front)
    for depth, expected in [(1.0, (np.log(2), 0)), (4.0, (0, np.log(2))), (2.0, (0, 0))]:
        actual = coverage_integrals(front, alpha, np.full_like(front, depth), alpha)
        assert_allclose(actual[:, 0, 0], expected, atol=1e-14)
    assert_allclose(
        coverage_integrals(front, alpha, np.full_like(front, 3000), alpha)[:, 0, 0],
        [0, np.log(100)],
        atol=1e-14,
    )


def test_pooled_regions_preserve_directions_and_ignore_other_contestants() -> None:
    inverse = np.full((2, 32, 32), 0.5)
    alpha = np.zeros_like(inverse)
    alpha[0] = 1
    target = Layers(inverse, alpha, np.zeros((2, 3, 32, 32)))
    prediction = Layers(inverse * 2, alpha.copy(), target.rgb)
    mixed = np.zeros((32, 32), dtype=bool)
    mixed[:, 16] = True
    samples = coverage_samples(prediction, target, mixed)
    first = coverage_summary(samples)
    bad = Layers(inverse * 0.001, alpha.copy(), target.rgb)
    coverage_samples(bad, target, mixed)
    assert coverage_summary(coverage_samples(prediction, target, mixed)) == first
    assert first["edge"]["pixels"] == 288
    assert first["interior"]["pixels"] == 736
    for region in ("edge", "interior"):
        assert first[region]["too_near"] == pytest.approx(np.log(2))
        assert first[region]["too_far"] == 0
    assert first["selection"] == pytest.approx(np.log(2))
    asymmetric = coverage_summary(
        {
            "edge_too_near": np.array([1.0]),
            "edge_too_far": np.array([2.0]),
            "interior_too_near": np.array([0.0, 0.0, 0.0]),
            "interior_too_far": np.array([1.0, 1.0, 1.0]),
        }
    )
    assert asymmetric["selection"] == 2.0
    for missing in ("edge", "interior"):
        incomplete = coverage_summary(
            {
                f"{region}_{direction}": np.array([] if region == missing else [1.0])
                for region in ("edge", "interior")
                for direction in ("too_near", "too_far")
            }
        )
        assert incomplete[missing]["pixels"] == 0
        assert incomplete["selection"] is None


def _metrics(sweep: float, leak: float, mixed: float = 0.1, step: int = 0) -> dict[str, Any]:
    return {
        "checkpoint_step": step,
        "coverage_sweep": {"selection": sweep},
        "leakage": {"passed": leak <= 0.001, "worst_mean": leak, "all": {"mean": leak}},
        "mixed_alpha_mae": mixed,
    }


def test_checkpoint_selection_is_order_independent_and_slot_mae_is_diagnostic() -> None:
    history = [
        _metrics(0.001, 0.1),
        _metrics(0.9, 0.01),
        _metrics(0.5, 0.0001),
        _metrics(0.2, 0.0009, 0.9),
    ]
    for stream in permutations(history):
        best = None
        for candidate in stream:
            if validation_improves(candidate, best):
                best = candidate
        assert best == history[3]
    for stream in permutations(history[:2]):
        best = None
        for candidate in stream:
            if validation_improves(candidate, best):
                best = candidate
        assert best == history[1]
    assert not validation_improves(history[3], history[3])
    for tied in (
        [_metrics(0.1, 0.0005, 0.9, 1), _metrics(0.1, 0.0006, 0.1, 0)],
        [_metrics(0.1, 0.0005, 0.9, 1), _metrics(0.1, 0.0005, 0.1, 2)],
    ):
        for first, second in permutations(tied):
            best = second if validation_improves(second, first) else first
            assert best == tied[0]


def test_sky_interior_is_scored_and_invalid_prediction_depth_is_not_sky() -> None:
    alpha = np.ones((1, 8, 8))
    target = Layers(np.zeros_like(alpha), alpha, np.zeros((1, 3, 8, 8)))
    prediction = Layers(np.full_like(alpha, 0.5), alpha, target.rgb)
    mixed = np.zeros((8, 8), dtype=bool)
    report = coverage_summary(coverage_samples(prediction, target, mixed))
    assert report["interior"]["pixels"] == 64
    assert report["interior"]["too_near"] == pytest.approx(np.log(100))
    assert report["interior"]["too_far"] == 0
    for invalid in (np.nan, np.inf, -1):
        prediction.inverse_depth[0, 0, 0] = invalid
        with pytest.raises(FloatingPointError, match="Invalid coverage-sweep inverse depth"):
            coverage_samples(prediction, target, mixed)


def test_prepass_prefers_overall_leakage_even_when_a_passing_bucket_rises() -> None:
    first, second = _metrics(0.3, 0.0015), _metrics(0.4, 0.0011)
    first["leakage"]["worst_mean"] = 0.05
    second["leakage"]["worst_mean"] = 0.06
    assert validation_improves(second, first)
    assert not validation_improves(first, second)


def test_opaque_interior_threshold_is_inclusive_0999() -> None:
    alpha = np.array([0.99, 0.9989, 0.999, 1.0])[None, None]
    target = Layers(np.ones_like(alpha), alpha, np.zeros((1, 3, 1, 4)))
    _, interior = coverage_regions(target, np.zeros((1, 4), dtype=bool))
    assert interior.tolist() == [[False, False, True, True]]


def test_point_deep_sweep_preserves_conditional_transmission() -> None:
    z = np.array([2.0, 4.0])[:, None, None]
    alpha = np.array([0.2, 1.0])[:, None, None]
    predicted_z = np.array([1.0, 8.0])[:, None, None]
    predicted_alpha = np.array([0.3, 1.0])[:, None, None]
    actual = coverage_integrals(z, alpha, predicted_z, predicted_alpha)
    assert_allclose(actual[:, 0, 0], [0.4 * np.log(2), 0.7 * np.log(2)], atol=1e-14)
