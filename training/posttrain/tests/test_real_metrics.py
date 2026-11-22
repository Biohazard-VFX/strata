import numpy as np
import pytest

from strata_training.eval.real.metrics import matte_metrics, temporal_error
from strata_training.eval.real.protocol import deep_to_matte


def test_front_slot_does_not_flatten_opaque_background() -> None:
    alpha = np.array([[[0.25]], [[1.0]], [[0.0]]])
    assert deep_to_matte(alpha).item() == 0.25
    membership = np.array([[[True]], [[False]], [[True]]])
    assert deep_to_matte(alpha, membership).item() == 0.25
    assert deep_to_matte(np.array([[[0.5]], [[0.5]]]), np.ones((2, 1, 1), bool)).item() == 0.75


def test_independent_absolute_squared_and_empty_region_values() -> None:
    gt = np.array([[0.0, 0.25, 0.75, 1.0]])
    pred = np.ones_like(gt)
    metrics = matte_metrics(pred, gt)
    assert metrics["mixed_mae"] == 0.5
    assert metrics["mixed_mse"] == 0.3125
    assert metrics["full_mae"] == 0.5
    assert metrics["sad"] == 0.002
    assert metrics["background_alpha"] == 1.0
    assert metrics["foreground_missing_alpha"] == 0.0
    assert matte_metrics(np.zeros((2, 2)), np.zeros((2, 2)))["mixed_mae"] is None


def test_gradient_and_connectivity_reject_missing_strand() -> None:
    gt = np.zeros((9, 9))
    gt[1:8, 4] = 1
    pred = gt.copy()
    pred[4, 4] = 0
    same = matte_metrics(gt, gt)
    broken = matte_metrics(pred, gt)
    assert same["grad"] == 0
    assert same["conn"] == 0
    assert (broken["grad"] or 0) > 0
    assert (broken["conn"] or 0) > 0
    assert (broken["boundary_mae"] or 0) > 0


def test_temporal_gt_residual_not_raw_flicker() -> None:
    zero = np.zeros((2, 2))
    one = np.ones((2, 2))
    assert temporal_error(one, zero, one, zero) == 0
    assert temporal_error(zero, zero, one, zero) == 100


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[1.01]])])
def test_invalid_mattes_fail_instead_of_silent_clipping(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"finite|range"):
        matte_metrics(bad, np.zeros_like(bad))
