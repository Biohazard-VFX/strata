import numpy as np
import pytest
from numpy.testing import assert_allclose

from strata_training.augment.grain import (
    FILM_RESPONSE,
    SHARPEN_KERNEL,
    GrainParams,
    apply_grain,
    grain_noise,
    response,
    sharpen,
)
from strata_training.augment.noise import GRADIENTS, gradient_noise, noise_node, octave_count

NUKE_RAMP = np.array(
    [
        *(-0.5, -0.17041015625, -0.0001220703125, 0.04931640625, 0.115234375),
        *(0.5986328125, 1.14794921875, 1.697265625, 3.89453125),
    ]
)
NUKE_FILM_RESPONSE = np.array(
    [
        [-1.34146, -1.781459, -0.792366],
        [-0.373196, -0.534247, -0.202005],
        [0.127074, 0.110146, 0.103015],
        [0.270593, 0.292228, 0.191122],
        [0.441477, 0.482892, 0.303935],
        [1.014711, 0.968919, 0.820985],
        [1.306305, 1.203394, 1.052925],
        [1.553497, 1.423121, 1.249427],
        [2.542267, 2.302027, 2.035438],
    ]
)


def test_gradient_noise_is_zero_on_lattice_with_cubic_fade_and_edge_gradients() -> None:
    lattice = np.array([0.0, 3.0, 17.0, 2047.0], dtype=np.float32)
    assert_allclose(gradient_noise(lattice, lattice[::-1], lattice), 0)
    zero = np.zeros(1, dtype=np.float32)

    def gradient(corner: float) -> list[float]:
        step = np.float32(1e-3)
        base = zero + np.float32(corner)
        return [
            float(gradient_noise(base + step * e[0], zero + step * e[1], zero + step * e[2])[0])
            / float(step)
            for e in np.eye(3, dtype=np.float32)
        ]

    g0, g1 = gradient(0.0), gradient(1.0)
    assert any(np.allclose(g0, g, atol=1e-2) for g in GRADIENTS)
    for t in (0.25, 0.7):
        fade = t * t * (3 - 2 * t)
        expected = (1 - fade) * round(g0[0]) * t + fade * round(g1[0]) * (t - 1)
        point = zero + np.float32(t)
        assert_allclose(gradient_noise(point, zero, zero), [expected], atol=1e-6)


def test_nyquist_octave_count_matches_nuke_noise_renders() -> None:
    observed = {0.9: 1, 1.1: 1, 1.55: 1, 2.0: 1, 2.2: 2, 4.0: 2, 4.8: 3, 8.0: 3, 9.0: 4, 50.0: 6}
    assert {size: octave_count(size) for size in observed} == observed


def test_noise_node_is_deterministic_and_centered() -> None:
    first = noise_node((64, 96), 4.0, 3.0)
    assert_allclose(first, noise_node((64, 96), 4.0, 3.0))
    other = noise_node((64, 96), 4.0, 4.0)
    assert np.abs(first - other).mean() > 0.05
    assert abs(float(noise_node((256, 256), 2.0, 7.0).mean()) - 0.5) < 0.01


def test_film_response_curves_match_nuke_color_lookup_with_linear_extrapolation() -> None:
    ours = np.stack([curve(NUKE_RAMP) for curve in FILM_RESPONSE], axis=1)
    assert_allclose(ours, NUKE_FILM_RESPONSE, atol=5e-6)


def test_response_brackets_the_curve_with_iso_exposure() -> None:
    image = np.full((3, 1, 1), 0.18, dtype=np.float32)
    stops = 1.0
    expected = [2.0**stops * curve(np.array(0.18 / 2.0**stops)) for curve in FILM_RESPONSE]
    assert_allclose(response(image, 1600)[:, 0, 0], expected, rtol=1e-6)


def test_sharpen_matches_nuke_impulse_response() -> None:
    impulse = np.zeros((17, 17), dtype=np.float32)
    impulse[8, 8] = 1
    result = sharpen(impulse, 0.25)
    assert_allclose(
        result[8, 6:11], [-0.007154, -0.018026, 1.225471, -0.018026, -0.007154], atol=2e-6
    )
    assert_allclose(result.sum(), 1, atol=1e-6)
    assert SHARPEN_KERNEL.size == 9


def test_grain_statistics_match_nuke_render_of_studio_group() -> None:
    noise = grain_noise((720, 1280), GrainParams("alexa_sxt", 800, 0.0, 777, 1.0))
    centred = noise - noise.mean(axis=(1, 2), keepdims=True)
    assert_allclose(centred.std(axis=(1, 2)), [0.01324, 0.01027, 0.0188], rtol=0.02)
    assert np.all(np.abs(noise.mean(axis=(1, 2))) < 0.02 * centred.std(axis=(1, 2)))
    lag = [(c[:, 1:] * c[:, :-1]).mean() / c.var() for c in centred]
    assert_allclose(lag, [0.53, 0.427, 0.576], atol=0.01)
    corr = np.atleast_2d(np.corrcoef(centred.reshape(3, -1)))
    assert_allclose([corr[0, 1], corr[0, 2], corr[1, 2]], [0.539, 0.398, 0.641], atol=0.01)


def test_grain_scales_with_amount_and_exposure_offset() -> None:
    image = np.full((3, 32, 48), 0.18, dtype=np.float32)
    base = GrainParams("alexa35", 800, 0.0, 5, 1.0)
    assert_allclose(apply_grain(image, GrainParams("alexa35", 800, 0.0, 5, 0.0)), image)
    single = apply_grain(image, base) - image
    double = apply_grain(image, GrainParams("alexa35", 800, 0.0, 5, 2.0)) - image
    assert_allclose(double, 2 * single, rtol=1e-4, atol=1e-7)
    under = apply_grain(image, GrainParams("alexa35", 800, 2.0, 5, 1.0)) - image
    assert_allclose(under, 2 * single, rtol=1e-4, atol=1e-7)
    clamped = apply_grain(image, GrainParams("alexa35", 800, 6.0, 5, 1.0)) - image
    assert_allclose(clamped, 3.25 * single, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("profile", ["alexa35", "alexa_lf", "alexa_sxt", "alexa65"])
def test_every_profile_produces_finite_signed_grain(profile: str) -> None:
    image = np.full((3, 48, 64), 0.18, dtype=np.float32)
    grain = apply_grain(image, GrainParams(profile, 3200, 0.0, 1, 1.0)) - image
    assert np.isfinite(grain).all()
    assert (grain > 0).any()
    assert (grain < 0).any()
