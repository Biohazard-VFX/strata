import numpy as np
import pytest
from numpy.testing import assert_allclose

from strata_training.augment.deep import flatten, from_unsorted, merge
from strata_training.augment.occluders import (
    KINDS,
    OCCLUDER_ID,
    Occluder,
    line_family,
    random_colour,
    random_shape,
    rasterize,
    samples,
    segment,
    tilted_depth,
)


@pytest.mark.parametrize("width", [0.15, 0.6, 1.0, 2.5])
@pytest.mark.parametrize("angle", [0.0, 0.3, -0.6, 1.2])
def test_thin_line_coverage_equals_analytic_area(width: float, angle: float) -> None:
    height, size = 300, 400
    steep = abs(np.sin(angle)) > abs(np.cos(angle))
    phase = 150.0 * np.cos(angle) - 200.0 * np.sin(angle)
    coverage = line_family(angle, width, 1e6, phase)((height, size))
    crossing = height / abs(np.sin(angle)) if steep else size / abs(np.cos(angle))
    assert_allclose(coverage.sum(), width * crossing, rtol=1e-9)


def test_periodic_lines_cover_their_duty_cycle() -> None:
    coverage = line_family(0.4, 1.3, 9.0, 2.0)((360, 360))
    assert_allclose(coverage.mean(), 1.3 / 9.0, rtol=0.01)


def test_finite_segment_area_and_partial_edges() -> None:
    start, end = np.array([40.3, 50.7]), np.array([170.9, 95.2])
    coverage = segment(start, end, 0.8)((160, 220))
    assert_allclose(coverage.sum(), 0.8 * np.hypot(*(end - start)), rtol=0.01)
    partial = (coverage > 0) & (coverage < 1)
    assert partial.sum() == (coverage > 0).sum()


def test_motion_blur_and_defocus_spread_but_preserve_coverage() -> None:
    shape = line_family(0.2, 2.0, 25.0, 3.0)
    sharp = rasterize(shape, (200, 200), (0.0, 0.0), 0.0)
    blurred = rasterize(shape, (200, 200), (9.0, 4.0), 3.0)
    inner = (slice(20, 180), slice(20, 180))
    assert_allclose(blurred[inner].mean(), sharp[inner].mean(), rtol=0.01)
    soft = [float(((c[inner] > 0.02) & (c[inner] < 0.98)).mean()) for c in (sharp, blurred)]
    assert soft[1] > 2 * soft[0]


@pytest.mark.parametrize("kind", KINDS)
def test_random_occluders_are_valid_and_seeded(kind: str) -> None:
    first = rasterize(
        random_shape(kind, (90, 160), np.random.default_rng(11)), (90, 160), (0, 0), 0
    )
    again = rasterize(
        random_shape(kind, (90, 160), np.random.default_rng(11)), (90, 160), (0, 0), 0
    )
    assert_allclose(first, again)
    assert first.min() >= 0
    assert first.max() <= 1
    assert first.max() > 0


def test_occluder_samples_merge_to_the_analytic_over_composite() -> None:
    rng = np.random.default_rng(12)
    size = (30, 40)
    coverage = rasterize(random_shape("fence", size, rng), size, (3.0, 0.0), 0.0)
    colour = random_colour(size, 0.2, True, rng)
    occluder = samples([Occluder(coverage, colour, tilted_depth(size, 0.5, rng))], size)
    assert set(occluder.ids.tolist()) == {OCCLUDER_ID}
    pixels = size[0] * size[1]
    background = from_unsorted(
        size,
        np.arange(pixels),
        np.full(pixels, 10.0, dtype=np.float32),
        np.ones(pixels, dtype=np.float32),
        np.tile(np.array([[0.1, 0.3, 0.5]], dtype=np.float32), (pixels, 1)),
        np.ones(pixels, dtype=np.int64),
    )
    merged, _ = merge([background, occluder])
    rgb, alpha = flatten(merged)
    expected = coverage * colour + (1 - coverage) * np.array([0.1, 0.3, 0.5])[:, None, None]
    assert_allclose(rgb, expected, rtol=1e-5, atol=1e-6)
    assert_allclose(alpha, 1, atol=1e-6)
    assert_allclose(merged.targets(2).alpha[0], np.where(coverage > 1e-6, coverage, 1), atol=1e-6)
