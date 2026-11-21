import numpy as np
import pytest
from numpy.testing import assert_allclose

from strata_training.data.targets import composite, reduce_samples


def test_conditional_layers_preserve_transmittance_color_and_weighted_depth() -> None:
    target = reduce_samples(
        np.array([12.0, 2.0, 4.0]),
        np.array([1.0, 0.5, 0.5]),
        np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        np.array([3, 0]),
        (1, 2),
        layers=4,
    )
    assert_allclose(target.alpha[:, 0, 0], [0.5, 0.5, 1, 0])
    assert_allclose(target.inverse_depth[:, 0, 0], [0.5, 0.25, 1 / 12, 0])
    assert_allclose(composite(target.alpha, target.rgb)[:, 0, 0], [0.5, 0.25, 0.25])
    assert not target.valid[:, 0, 1].any()
    merged = reduce_samples(
        np.array([2.0, 4.0, 12.0]),
        np.array([0.5, 0.5, 1.0]),
        np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]),
        np.array([3]),
        (1, 1),
        layers=2,
    )
    assert_allclose(merged.alpha[:, 0, 0], [0.75, 1])
    assert_allclose(merged.inverse_depth[:, 0, 0], [(0.5 * 0.5 + 0.25 * 0.25) / 0.75, 1 / 12])
    assert_allclose(composite(merged.alpha, merged.rgb)[:, 0, 0], [0.5, 0.25, 0.25])


def test_opaque_samples_hide_later_samples_and_small_gaps_do_not_split() -> None:
    target = reduce_samples(
        np.array([2.0, 2.1, 10.0]),
        np.array([0.5, 1.0, 1.0]),
        np.array([[0.5, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([3]),
        (1, 1),
        layers=4,
    )
    assert_allclose(target.alpha[:, 0, 0], [1, 0, 0, 0], atol=1e-10)
    assert_allclose(composite(target.alpha, target.rgb)[:, 0, 0], [0.5, 0.5, 0])


def test_hidden_depth_gaps_do_not_consume_visible_layer_slots() -> None:
    target = reduce_samples(
        np.array([2.0, 4.0, 8.0, 100.0, 10000.0]),
        np.array([0.5, 0.5, 1.0, 1.0, 1.0]),
        np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]]),
        np.array([5]),
        (1, 1),
        layers=3,
    )
    assert_allclose(target.alpha[:, 0, 0], [0.5, 0.5, 1])
    assert_allclose(target.inverse_depth[:, 0, 0], [0.5, 0.25, 0.125])


def test_invalid_deep_samples_fail_instead_of_silently_reordering_bad_data() -> None:
    with pytest.raises(ValueError, match="positive"):
        reduce_samples(np.array([0.0]), np.ones(1), np.zeros((1, 3)), np.ones(1, dtype=int), (1, 1))
