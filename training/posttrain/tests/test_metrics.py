import numpy as np
from numpy.testing import assert_allclose

from strata_training.eval.metrics import boundary_recall, mixed_errors


def test_mixed_metrics_exclude_unmixed_pixels_and_use_actual_back_layer() -> None:
    pred = np.array([[[0.5, 99]], [[0.3, 99]], [[0.1, 99]], [[0, 99]]])
    truth = np.array([[[0.4, 1]], [[0.2, 0]], [[0.1, 0]], [[0, 0]]])
    alpha = np.array([[[0.5, 0.1]], [[1e-8, 1]], [[1, 0]], [[0, 0]]])
    errors = mixed_errors(pred, np.full((1, 2), 0.6), truth, alpha, np.array([[True, False]]))
    assert_allclose(errors[0], [0.1])
    assert_allclose(errors[1], [0.25])
    assert_allclose(errors[2], [0])


def test_boundary_recall_counts_matched_gt_edges_and_penalizes_missing_edges() -> None:
    truth = np.ones((7, 9), dtype=np.float32)
    truth[:, 5:] = 2
    assert boundary_recall(truth, truth, ratio=1.25, tolerance=0) == 1.0
    assert boundary_recall(np.ones_like(truth), truth, ratio=1.25, tolerance=1) == 0.0
    shifted = np.ones_like(truth)
    shifted[:, 6:] = 2
    assert boundary_recall(shifted, truth, ratio=1.25, tolerance=1) == 1.0
