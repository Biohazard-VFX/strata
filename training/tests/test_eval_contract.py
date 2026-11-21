from pathlib import Path

import numpy as np
import torch
from numpy.testing import assert_allclose

from strata_training.data.exr import read_deep, write_deep
from strata_training.eval.boundaries import boundary_f1, directed_boundary_counts, silhouette_recall
from strata_training.eval.calibration import calibration, tracked_depth_jitter
from strata_training.losses.deep import gradient_loss
from strata_training.types import Layers


def test_directed_boundary_detects_sign_and_location_not_global_scale() -> None:
    inverse = np.ones((9, 11))
    inverse[:, 5:] = 0.5
    assert boundary_f1(directed_boundary_counts(inverse * 3, inverse)) == 1
    assert boundary_f1(directed_boundary_counts(1 / inverse, inverse)) == 0
    assert boundary_f1(directed_boundary_counts(np.roll(inverse, 2, axis=1), inverse)) == 0
    silhouette = inverse > 0.75
    assert silhouette_recall(inverse, silhouette) == 1
    assert silhouette_recall(np.ones_like(inverse), silhouette) == 0


def test_alpha_calibration_and_track_jitter_are_not_accuracy_substitutes() -> None:
    report = calibration(np.array([0.1, 0.1, 0.9, 0.9]), np.array([0.2, 0.2, 0.8, 0.8]))
    assert_allclose(report["ece"], 0.1)
    assert_allclose(report["brier"], 0.01)
    truth = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    valid = np.ones_like(truth, dtype=bool)
    steady = tracked_depth_jitter(truth * 2, truth, valid)
    assert steady is not None
    assert_allclose(steady, 0, atol=1e-12)
    prediction = truth * np.array([[1], [2], [1]])
    jitter = tracked_depth_jitter(prediction, truth, valid)
    assert jitter is not None
    assert_allclose(jitter, np.log(2))


def test_three_native_stride_gradients_ignore_offsets_not_edges() -> None:
    truth = torch.ones((1, 4, 16, 16))
    valid = torch.ones_like(truth, dtype=torch.bool)
    assert gradient_loss(truth + 1, truth, valid).item() == 0
    step = truth.clone()
    step[..., 8:] += 1
    assert gradient_loss(step, truth, valid).item() > 0


def test_deep_exr_round_trip_retains_conditional_colors_and_empty_pixels(tmp_path: Path) -> None:
    inverse = np.array([[[0.5, 0]], [[0.25, 0]]], dtype=np.float32)
    alpha = np.array([[[0.5, 0]], [[1, 0]]], dtype=np.float32)
    rgb = np.array(
        [[[[0.5, 0]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, 0]]]], dtype=np.float32
    )
    path = tmp_path / "deep.exr"
    write_deep(path, Layers(inverse, alpha, rgb))
    restored = read_deep(path, layers=2)
    assert_allclose(restored.inverse_depth, inverse)
    assert_allclose(restored.alpha, alpha)
    assert_allclose(restored.rgb, rgb)
