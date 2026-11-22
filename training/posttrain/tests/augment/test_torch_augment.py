from dataclasses import replace

import numpy as np
import pytest
import torch

from strata_training.augment import camera, grain, occluders
from strata_training.augment.deep import Shot, from_unsorted
from strata_training.augment.pipeline import AugmentConfig, Donor, Roles, recompose
from strata_training.augment.torch_compose import TensorShot
from strata_training.augment.torch_compose import recompose as tensor_recompose
from strata_training.augment.torch_deep import TensorSamples
from strata_training.augment.torch_grain import apply_grain
from strata_training.augment.torch_occluders import random_shape, rasterize


@pytest.mark.parametrize("codec", camera.CODECS)
@pytest.mark.parametrize("curve", camera.CURVES)
def test_torch_camera_matches_numpy(codec: str, curve: str, math_device: str) -> None:
    from strata_training.augment.torch_camera import degrade

    image = np.random.default_rng(199).uniform(0.01, 2, (3, 32, 48)).astype(np.float32)
    params = camera.CameraParams(1.1, 0.9, curve, 8, 0.3, 1.2, codec, 33, 8)
    expected = camera.degrade(image, params)
    actual = degrade(torch.tensor(image, device=math_device), params).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=2e-5)


def test_torch_batched_grain_matches_independent_cpu_fields(math_device: str) -> None:
    from strata_training.augment.torch_grain import apply_grain_batch

    images = np.random.default_rng(44).uniform(-0.02, 8, (3, 3, 64, 96)).astype(np.float32)
    params = [
        grain.GrainParams("alexa35", 400, -1.1, 13, 0.7),
        grain.GrainParams("alexa_lf", 3200, 0.9, 71, 1.3),
        grain.GrainParams("alexa35", 1280, 0.3, 106, 1.0),
    ]
    expected = np.stack(
        [grain.apply_grain(image, p) for image, p in zip(images, params, strict=True)]
    )
    actual = apply_grain_batch(torch.tensor(images, device=math_device), params)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("profile", list(grain.PROFILES))
def test_torch_grain_matches_numpy(profile: str, math_device: str) -> None:
    image = np.random.default_rng(42).uniform(-0.01, 8, (3, 31, 43)).astype(np.float32)
    params = grain.GrainParams(profile, 1280, -1.3, 1901, 1.2)
    expected = grain.apply_grain(image, params)
    actual = apply_grain(torch.tensor(image, device=math_device), params).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_noise_replays_across_layouts_and_grad_modes(math_device: str) -> None:
    from strata_training.augment.noise import gradient_noise as reference_noise
    from strata_training.augment.torch_grain import gradient_noise

    rng = np.random.default_rng(431)
    for dtype in (np.float32, np.float64):
        for shape in ((31, 43), (1, 31, 43), (3, 31, 43)):
            for enabled in (False, True):
                for transpose in (False, True):
                    arrays = [rng.uniform(-5, 5, shape).astype(dtype) for _ in range(3)]
                    values = [torch.tensor(a, device=math_device) for a in arrays]
                    if transpose:
                        values = [value.transpose(-2, -1) for value in values]
                        arrays = [np.swapaxes(a, -2, -1) for a in arrays]
                    with torch.set_grad_enabled(enabled):
                        actual = gradient_noise(*values).cpu().numpy()
                    np.testing.assert_allclose(
                        actual, reference_noise(*arrays), atol=2e-6, rtol=2e-6
                    )


@pytest.mark.parametrize("kind", occluders.KINDS)
def test_torch_occluder_matches_numpy(kind: str, math_device: str) -> None:
    size = (31, 43)
    motion, defocus = (3.3, -2.7), 2.2
    expected = occluders.rasterize(
        occluders.random_shape(kind, size, np.random.default_rng(123)), size, motion, defocus
    )
    actual = (
        rasterize(
            random_shape(kind, size, np.random.default_rng(123), torch.device(math_device)),
            size,
            motion,
            defocus,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("operation", ["scene_grade", "instance_grade", "recompose", "occluders"])
def test_torch_recomposition_matches_cpu(operation: str, math_device: str) -> None:
    shape = (12, 14)
    rng = np.random.default_rng(1901)
    pixels = np.repeat(np.arange(np.prod(shape)), 5)
    z = rng.uniform(1, 20, pixels.size).astype(np.float32)
    alpha = rng.uniform(0.05, 0.9, pixels.size).astype(np.float32)
    rgb = rng.uniform(0.01, 0.4, (pixels.size, 3)).astype(np.float32) * alpha[:, None]
    ids = rng.integers(0, 5, pixels.size)
    deep = from_unsorted(shape, pixels, z, alpha, rgb, ids)
    noisy = rng.uniform(0.01, 1, (3, *shape)).astype(np.float32)
    shot = Shot(deep, noisy, noisy * 0.97)
    donor = Shot(deep.scaled(0.7), noisy * 1.3, noisy * 1.25)
    roles = Roles(frozenset([2, 3]), ((4, 1),))
    config = replace(
        AugmentConfig(scene_grade=0, instance_grade=0, recompose=0, occluders=0), **{operation: 1}
    )
    expected_record: dict[str, object] = {}
    target = deep.targets()
    expected_image, expected = recompose(
        shot, roles, target, Donor(donor, roles), np.random.default_rng(42), config, expected_record
    )

    def tensor_shot(value: Shot) -> TensorShot:
        d = value.deep
        return TensorShot(
            TensorSamples(
                d.shape,
                *(
                    torch.tensor(v, device=math_device)
                    for v in (d.counts, d.z, d.alpha, d.rgb, d.ids)
                ),
            ),
            torch.tensor(value.noisy, device=math_device),
            torch.tensor(value.denoised, device=math_device),
        )

    actual_record: dict[str, object] = {}
    image, actual = tensor_recompose(
        tensor_shot(shot),
        roles,
        {k: torch.tensor(v, device=math_device) for k, v in target.arrays().items()},
        tensor_shot(donor),
        roles,
        np.random.default_rng(42),
        config,
        actual_record,
    )
    assert actual_record.keys() == expected_record.keys()
    np.testing.assert_allclose(image.cpu().numpy(), expected_image, atol=2e-6, rtol=2e-6)
    for key in ("inverse_depth", "alpha", "rgb"):
        np.testing.assert_allclose(
            actual[key].cpu().numpy(), getattr(expected, key), atol=2e-6, rtol=2e-6
        )
