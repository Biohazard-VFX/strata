import numpy as np
import pytest
from numpy.testing import assert_allclose

from strata_training.eval.leakage import leakage_samples, leakage_summary, validation_improves
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
    assert empty["all"] == {"mean": None, "pixels": 0}
    assert not empty["passed"]


def test_leakage_gate_and_selection_require_both_metrics() -> None:
    def metrics(mixed: float, leak: float) -> dict[str, object]:
        return {
            "mixed_alpha_mae": mixed,
            "leakage": leakage_summary({"all": np.array([leak]), "dark": np.array([leak])}),
        }

    best = metrics(0.2, 0.1)
    assert not validation_improves(metrics(0.1, 0.2), best)
    assert not validation_improves(metrics(0.3, 0.05), best)
    assert validation_improves(metrics(0.1, 0.05), best)
    assert validation_improves(metrics(0.2, 0.05), best)
    assert validation_improves(metrics(0.9, 0.001), best)
    assert not validation_improves(best, metrics(0.9, 0.001))
    assert not validation_improves(best, best)
    assert validation_improves(best, None)
    assert not leakage_summary({"all": np.array([0.0005]), "bright": np.array([0.002])})["passed"]
