import numpy as np
import pytest
from numpy.testing import assert_allclose

from strata_training.eval.leakage import leakage_samples, leakage_summary
from strata_training.types import Layers


def _layers(alpha: float, inverse: float = 0.5, size: int = 16) -> Layers:
    return Layers(
        np.full((1, size, size), inverse),
        np.full((1, size, size), alpha, dtype=np.float64),
        np.zeros((1, 3, size, size)),
    )


@pytest.mark.parametrize(
    ("alpha", "inverse", "expected"),
    [(1, 0.5, 0), (0.5, 0.5, 0.5), (0, 0.5, 1), (0, 0, 1), (1, 0.25, 1)],
)
def test_behind_surface_leakage(alpha: float, inverse: float, expected: float) -> None:
    beauty = np.tile(np.array([0.01, 0.5, 2, 0.5]), (3, 16, 4))
    samples = leakage_samples(_layers(alpha, inverse), _layers(1), np.zeros((16, 16)), beauty)
    report = leakage_summary(samples)
    for region in ("all", "bright", "normal", "dark"):
        assert report[region]["pixels"] > 0
        assert report[region]["mean"] == expected
    assert report["passed"] == (expected == 0)


@pytest.mark.parametrize(("depth", "offset"), [(0.5, 0.01), (2.0, 0.02)])
def test_plane_uses_one_centimetre_or_one_percent(depth: float, offset: float) -> None:
    target = _layers(1, 1 / depth)
    for distance, expected in [(offset * 0.99, 0), (offset * 1.01, 1)]:
        prediction = _layers(1, 1 / (depth + distance))
        samples = leakage_samples(prediction, target, np.zeros((16, 16)), np.zeros((3, 16, 16)))
        assert leakage_summary(samples)["all"]["mean"] == expected


def test_leakage_uses_gt_plane_opacity_and_interior_not_prediction_mask() -> None:
    gt = _layers(1)
    gt.alpha[0, 0, 0] = 0.96
    mixed = np.zeros((16, 16), dtype=bool)
    mixed[:, 8] = True
    samples = leakage_samples(_layers(0.5), gt, mixed, np.zeros((3, 16, 16)))
    assert samples["all"].size == 111
    assert_allclose(samples["all"], 0.5)
    empty = leakage_summary(leakage_samples(_layers(1), _layers(0), mixed, np.zeros((3, 16, 16))))
    assert empty["all"]["mean"] is None
    assert empty["all"]["pixels"] == 0
    assert empty["all"]["status"] == "not evaluated"
    assert not empty["passed"]


@pytest.mark.parametrize(
    ("count", "mean", "status"),
    [
        (0, 0.9, "not evaluated"),
        (249, 0.9, "not evaluated"),
        (250, 0.1, "pass"),
        (250, 0.10001, "fail"),
        (1000, 0.025, "pass"),
        (1000, 0.02501, "fail"),
        (25000, 0.001, "pass"),
        (25000, 0.00101, "fail"),
        (50000, 0.00101, "fail"),
    ],
)
def test_count_aware_brightness_gate(count: int, mean: float, status: str) -> None:
    report = leakage_summary({"all": np.array([0.0005]), "bright": np.full(count, mean)})
    assert report["bright"]["status"] == status
    assert report["passed"] == (status != "fail")
    assert not leakage_summary({"all": np.array([0.00101]), "bright": np.zeros(count)})["passed"]


def test_bright_gate_pools_pixels_before_count_threshold() -> None:
    frames = [np.full(125, 0.2), np.zeros(125)]
    for frame in frames:
        assert (
            leakage_summary({"all": np.array([0.0]), "bright": frame})["bright"]["status"]
            == "not evaluated"
        )
    report = leakage_summary({"all": np.array([0.0]), "bright": np.concatenate(frames)})
    assert report["bright"]["status"] == "pass"
    assert report["bright"]["mean"] == pytest.approx(0.1)


def test_premature_occlusion_is_report_only_and_gt_surface_is_visible() -> None:
    mixed, beauty = np.zeros((16, 16)), np.zeros((3, 16, 16))
    for depth, expected in [(1.0, 1.0), (2.0, 0.0), (4.0, 0.0)]:
        report = leakage_summary(leakage_samples(_layers(1, 1 / depth), _layers(1), mixed, beauty))
        assert report["premature"]["mean"] == expected
        assert report["premature"]["status"] == "not evaluated"
        assert report["passed"] == (depth <= 2)
