import numpy as np
import torch
from strata_contract.layers import reduce_samples

from strata_training.augment.torch_deep import TensorSamples


def test_torch_targets_match_contract(math_device: str) -> None:
    rng = np.random.default_rng(1901)
    counts = rng.integers(0, 32, 120, dtype=np.int64)
    count = int(counts.sum())
    z = np.exp(rng.uniform(-2, 5, count)).astype(np.float32)
    alpha = rng.uniform(0, 1, count).astype(np.float32)
    alpha[::7] = 1
    alpha[::13] = 0
    rgb = rng.uniform(-1, 4, (count, 3)).astype(np.float32) * alpha[:, None]
    ids = rng.integers(0, 4, count, dtype=np.int64)
    expected = reduce_samples(
        z.astype(np.float64),
        alpha.astype(np.float64),
        rgb.astype(np.float64),
        counts,
        (10, 12),
        sky=ids == 0,
    )
    pixel = np.repeat(np.arange(counts.size), counts)
    order = np.lexsort((z, pixel))
    samples = TensorSamples(
        (10, 12),
        torch.tensor(counts, device=math_device),
        torch.tensor(z[order], device=math_device),
        torch.tensor(alpha[order], device=math_device),
        torch.tensor(rgb[order], device=math_device),
        torch.tensor(ids[order], device=math_device),
    )
    actual = samples.targets()
    for key in ("inverse_depth", "alpha", "rgb"):
        np.testing.assert_allclose(
            actual[key].cpu().numpy(), getattr(expected, key), atol=2e-6, rtol=2e-6
        )
    from dataclasses import replace

    from strata_training.augment.torch_deep import batch_samples

    other = replace(
        samples, z=samples.z * 3, rgb=samples.rgb * 0.4, ids=torch.ones_like(samples.ids)
    )
    second = reduce_samples(
        (z[order] * 3).astype(np.float64),
        alpha[order].astype(np.float64),
        (rgb[order] * 0.4).astype(np.float64),
        counts,
        (10, 12),
    )
    packed = batch_samples([samples, other]).targets()
    for key in ("inverse_depth", "alpha", "rgb"):
        expected_batch = np.concatenate((getattr(expected, key), getattr(second, key)), axis=-2)
        np.testing.assert_allclose(packed[key].cpu().numpy(), expected_batch, atol=2e-6, rtol=2e-6)


def test_torch_targets_hidden_sky_ties_empty(math_device: str) -> None:
    counts = np.array([0, 6, 3, 2], dtype=np.int64)
    z = np.array([1, 2, 4, 8, 16, 32, 1, 2, 3, 1, 2], dtype=np.float32)
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1, 1, 0.5, 1, 0, 0], dtype=np.float32)
    rgb = np.repeat(alpha[:, None], 3, axis=1)
    ids = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    expected = reduce_samples(
        z.astype(np.float64),
        alpha.astype(np.float64),
        rgb.astype(np.float64),
        counts,
        (2, 2),
        sky=ids == 0,
    )
    samples = TensorSamples(
        (2, 2), *(torch.tensor(v, device=math_device) for v in (counts, z, alpha, rgb, ids))
    )
    actual = samples.targets()
    for key in ("inverse_depth", "alpha", "rgb"):
        np.testing.assert_allclose(
            actual[key].cpu().numpy(), getattr(expected, key), atol=2e-6, rtol=2e-6
        )
