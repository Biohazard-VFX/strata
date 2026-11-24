import numpy as np
import torch
from numpy.testing import assert_allclose

from strata_training.baselines.deep import fake_deep, two_layer
from strata_training.eval.boundaries import boundary_f1, directed_boundary_counts, silhouette_recall
from strata_training.eval.calibration import calibration, tracked_depth_jitter
from strata_training.eval.runner import Collected, collect
from strata_training.losses.deep import gradient_loss
from strata_training.types import Layers


def test_zero_analytic_foreground_coverage_is_not_relabelled_as_opaque() -> None:
    shape = (16, 16)
    beauty = np.full((3, *shape), 0.5)
    target = two_layer(np.full(shape, 0.2), np.full(shape, 0.5), np.full(shape, 0.1), beauty)
    prediction = two_layer(np.zeros(shape), np.full(shape, 0.5), np.full(shape, 0.1), beauty)
    data = {
        "mixed": np.ones(shape, dtype=bool),
        "fur": np.zeros(shape, dtype=bool),
        "foreground": np.ones(shape, dtype=np.float32),
        "beauty": beauty,
    }
    results: dict[tuple[str, str], Collected] = {}
    collect(prediction, target, data, "analytic r=6 raw metric", {"category": "sharp"}, results)
    summary = results["all", "analytic r=6 raw metric"].summary()
    assert_allclose(summary["alpha_mae"], 0.2)
    assert summary["front_median_relative_error"] == 0
    assert summary["back_median_relative_error"] == 0


def test_solid_frames_remain_in_geometry_metrics_without_fabricated_alpha_scores() -> None:
    target = Layers(np.zeros((4, 16, 16)), np.zeros((4, 16, 16)), np.zeros((4, 3, 16, 16)))
    target.inverse_depth[0] = 0.5
    target.alpha[0] = 1
    target.rgb[0] = 0.5
    beauty = np.full((3, 16, 16), 0.5)
    prediction = fake_deep(np.full((16, 16), 0.25), beauty)
    data = {
        "mixed": np.zeros((16, 16), dtype=bool),
        "fur": np.zeros((16, 16), dtype=bool),
        "foreground": np.ones((16, 16), dtype=np.float32),
        "beauty": beauty,
    }
    results: dict[tuple[str, str], Collected] = {}
    collect(prediction, target, data, "fake", {"category": "sharp"}, results)
    summary = results["all", "fake"].summary()
    assert summary["frames"] == 1
    assert summary["mixed_pixels"] == 0
    assert summary["alpha_mae"] is None
    assert summary["coverage_calibration"]["ece"] is None
    assert summary["depth_absrel"] == 1
    assert not any("align" in key for key in summary)
    assert summary["depth_delta1"] == 0
    assert summary["composite_mae"] == 0
    assert summary["leakage"]["all"]["mean"] == 1
    assert summary["leakage"]["all"]["pixels"] == 256
    assert not summary["leakage"]["passed"]
    target.alpha[:, 0, 0] = 0
    target.inverse_depth[:, 0, 0] = 0
    target.rgb[:, :, 0, 0] = 0
    beauty[:, 0, 0] = 0
    matched = fake_deep(np.full((16, 16), 0.5), beauty)
    collect(matched, target, data, "matched", {"category": "sharp"}, results)
    assert results["all", "matched"].summary()["depth_absrel"] == 0


def test_evaluation_uses_fractional_stored_foreground_beyond_three_metres() -> None:
    beauty = np.full((3, 8, 12), 0.5)
    target = fake_deep(np.full((8, 12), 0.01), beauty)
    inverse = np.full((8, 12), 0.1)
    inverse[:, 6:] = 0.05
    prediction = fake_deep(inverse, beauty)
    foreground = np.full((8, 12), 0.5)
    foreground[:, 6:] = 0.05
    data = {
        "mixed": np.zeros((8, 12), dtype=bool),
        "beauty": beauty,
        "foreground": foreground,
        "fur": np.where(foreground > 0.1, 0.025, 0).astype(np.float32),
    }
    results: dict[tuple[str, str], Collected] = {}
    collect(prediction, target, data, "stored", {"category": "sharp"}, results)
    assert results["all", "stored"].summary()["silhouette_recall_foreground"] == 1
    assert results["sharp / fur", "stored"].pixels == 48
    assert results["sharp / hard", "stored"].pixels == 48


def test_leakage_regions_and_visible_sample_counts_are_pooled_independently() -> None:
    alpha = np.zeros((4, 16, 16))
    inverse = np.full_like(alpha, 0.5)
    alpha[0] = 0.99
    alpha[1] = 1
    target = Layers(inverse, alpha, np.zeros((4, 3, 16, 16)))
    prediction = Layers(inverse.copy(), alpha.copy(), target.rgb.copy())
    prediction.alpha[0, :, :8] = 1
    prediction.alpha[1:, :, :8] = 1
    prediction.inverse_depth[:, :, 8:] = 0.25
    fur = np.zeros((16, 16))
    fur[:, :8] = 1
    data = {
        "mixed": np.zeros((16, 16), dtype=bool),
        "fur": fur,
        "foreground": np.ones((16, 16)),
        "beauty": np.zeros((3, 16, 16)),
    }
    results: dict[tuple[str, str], Collected] = {}
    collect(prediction, target, data, "test", {"category": "sharp"}, results)
    for group, leakage, count in [
        ("all", 0.5, 256),
        ("sharp / fur", 0.0, 128),
        ("sharp / hard", 1.0, 128),
    ]:
        report = results[group, "test"].summary()
        assert report["leakage"]["all"]["mean"] == leakage
        assert report["leakage"]["all"]["pixels"] == count
        assert report["sample_count_mae_a003"] == 0


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
    assert report["ece"] is not None
    assert_allclose(report["ece"], 0.1)
    assert report["brier"] is not None
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
