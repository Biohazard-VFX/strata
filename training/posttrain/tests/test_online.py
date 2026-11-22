from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import authored_roles
from torch import nn

from strata_training.augment.deep import DeepSamples, save_samples
from strata_training.config import training_defaults, validate_config
from strata_training.data.dataset import Frame
from strata_training.data.online import OnlineLoader, image_digest
from strata_training.io import save_array, save_arrays, write_json
from strata_training.model.live import FrozenEncoder
from strata_training.torch_api import api
from strata_training.types import Array


def test_frozen_encoder_has_independent_batches_and_trainable_consumers() -> None:
    upstream: Any = import_module("depth_pro_2.model")
    config = upstream.PRESETS["depthpro2-s"]
    api.manual_seed(719)
    base: Any = (
        upstream.MultiViewEncoder(
            backbone_preset=config.backbone_preset,
            reassembled_blocks=config.reassembled_blocks,
            cross_frame_blocks=config.cross_frame_blocks,
            dims_reassemble=config.dims_reassemble,
        )
        .eval()
        .requires_grad_(False)
    )
    for name, parameter in base.named_parameters():
        if name.endswith((".gamma_1", ".gamma_2")):
            parameter.fill_(1)
    encoder = FrozenEncoder(base)
    image = torch.randn(2, 3, 32, 64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        tokens, grid = base.encode(image[:1])
        reference = base.reassemble_tokens(tokens, grid)
    for expected, actual in zip(reference, encoder(image[:1]), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original = encoder(image)
    changed = image.clone()
    changed[1] *= -7
    for expected, actual in zip(original, encoder(changed), strict=True):
        torch.testing.assert_close(actual[:1], expected[:1], rtol=0, atol=0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        a, grid = base.encode(image)
        b, _ = base.encode(changed)
        joint_a = base.reassemble_tokens(a, grid)
        joint_b = base.reassemble_tokens(b, grid)
    assert any(not torch.equal(a[:1], b[:1]) for a, b in zip(joint_a, joint_b, strict=True))
    head = nn.Conv2d(original[0].shape[1], 1, 1)
    api.autograd.backward(head(original[0].float()).square().mean())
    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for parameter in base.parameters())


@pytest.mark.parametrize("loader_math", ["numpy", "torch"])
def test_online_native_crops_use_denoised_inputs_and_reject_stale_clean_features(
    tmp_path: Path,
    loader_math: str,
    math_device: str,
) -> None:
    height, width = 32, 64
    image = np.full((3, height, width), 0.5, dtype=np.float32)
    target = np.zeros((4, height, width), dtype=np.float32)
    target[:2] = np.array([0.2, 1.0])[:, None, None]
    inverse = target.copy()
    inverse[:2] = np.array([0.5, 0.25])[:, None, None]
    colour = np.full((4, 3, height, width), 0.1, dtype=np.float32)
    cache = tmp_path / "target.npz"
    save_arrays(
        cache,
        alpha=target,
        inverse_depth=inverse,
        rgb=colour,
        beauty=image * 1.6,
        target_beauty=image * 1.4,
        mixed=np.ones((height, width), dtype=bool),
    )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"instance_ids": []})
    root = tmp_path / "features"
    save_array(root / "sample/features.npy", np.zeros((128, height, width), dtype=np.float32))
    for level in range(4):
        save_array(root / f"sample/pyramid_{level}.npy", np.zeros((1, 1, 1)))
    save_arrays(
        root / "sample/base.npz", raw_inverse_depth=np.ones((height, width)), raw_metric_scale=1.0
    )
    write_json(root / "sample/context.json", {"image_sha256": image_digest(image)})
    save_array(root / "sample/image.npy", image)
    samples = tmp_path / "samples"
    data = DeepSamples(
        (height, width),
        np.arange(height * width + 1) * 2,
        np.tile([2.0, 4.0], height * width).astype(np.float32),
        (fixture_alpha := np.tile([0.2, 1.0], height * width).astype(np.float32)),
        np.full((height * width * 2, 3), 0.1, dtype=np.float32),
        (fixture_ids := np.tile([2, 0], height * width).astype(np.int64)),
        authored_roles(fixture_alpha, fixture_ids),
    )
    save_samples(samples / "sample/deep", data)
    save_array(samples / "sample/noisy.npy", image * 1.4)
    save_array(samples / "sample/denoised.npy", image)
    frame = Frame(
        {
            "id": "sample",
            "cache": str(cache),
            "manifest": str(manifest),
            "recomposition_eligible": True,
        },
        root,
    )
    np.testing.assert_array_equal(frame.arrays["beauty"], image)
    loader = OnlineLoader([frame], samples, 1)
    clean = loader.prepare([(0, 0, 0)], np.random.default_rng(31), 32, 0, "cpu")
    assert clean.clean == [0]
    assert clean.live == []
    assert clean.normalized is None
    torch.testing.assert_close(clean.images, torch.full((1, 3, 32, 32), 0.5))
    torch.testing.assert_close(clean.target["beauty"], torch.full((1, 3, 32, 32), 0.7))
    assert clean.scales.item() == 1
    loader = OnlineLoader(
        [frame],
        samples,
        0,
        {
            "recompose": 0,
            "occluders": 0,
            "instance_grade": 0,
            "scene_grade": 1,
            "exposure_stops": [1, 1],
            "white_balance_stops": 0,
            "grain": 0,
            "camera": 0,
        },
    )
    changed = loader.prepare([(0, 0, 0)], np.random.default_rng(31), 32, 0, "cpu")
    assert changed.live == [0]
    assert changed.clean == []
    assert changed.normalized is not None
    torch.testing.assert_close(changed.images, torch.ones((1, 3, 32, 32)))
    torch.testing.assert_close(changed.target["beauty"], torch.full((1, 3, 32, 32), 0.36))
    torch.testing.assert_close(changed.target["alpha"], torch.as_tensor(target[:, :, :32])[None])
    torch.testing.assert_close(changed.target["rgb"], torch.full((1, 4, 3, 32, 32), 0.2))
    assert changed.scales.item() == 0.5
    from strata_training.data.workers import LoadedBatch, NativeBatches, batches

    config: dict[str, Any] = {
        "seed": 311,
        "loader_math": loader_math,
        "crop": 32,
        "halo": 0,
        "batch_size": 2,
        "steps": 6,
        "mode": "head",
        "device": "cpu",
        "loader_workers": 0,
        "loader_prefetch": 2,
    }
    online = OnlineLoader(
        [frame],
        samples,
        0.4,
        {
            "recompose": 0,
            "occluders": 0,
            "instance_grade": 1,
            "scene_grade": 1,
            "grain": 0,
            "camera": 0,
        },
    )
    sequential: list[LoadedBatch] = [
        row.to("cpu") for row in batches(NativeBatches([frame], online, [[0]], config))
    ]
    parallel: list[LoadedBatch] = list(
        batches(NativeBatches([frame], online, [[0]], {**config, "loader_workers": 2}))
    )
    resumed: list[LoadedBatch] = list(batches(NativeBatches([frame], online, [[0]], config, 3)))
    assert len({float(row.prepared.images.mean()) for row in sequential if row.prepared}) > 1
    for expected, actual in [
        *zip(sequential, parallel, strict=True),
        *zip(sequential[3:], resumed, strict=True),
    ]:
        a, b = expected.prepared, actual.to("cpu").prepared
        assert a is not None
        assert b is not None
        assert (a.clean, a.live, a.selections) == (b.clean, b.live, b.selections)
        torch.testing.assert_close(a.images, b.images, rtol=0, atol=0)
        for key, value in a.target.items():
            torch.testing.assert_close(value, b.target[key], rtol=0, atol=0)
        for key, value in (a.cached_inputs or {}).items():
            assert b.cached_inputs is not None
            torch.testing.assert_close(value, b.cached_inputs[key], rtol=0, atol=0)
    if math_device == "cuda":
        from strata_training.data.workers import device_batches

        cuda_config = {**config, "device": "cuda"}
        pinned = list(batches(NativeBatches([frame], online, [[0]], cuda_config)))
        for row in pinned:
            assert row.prepared is not None
            assert row.prepared.images.is_pinned()
            for job in row.prepared.jobs:
                if job.shot.deep is not None:
                    for channel in ("counts", "z", "alpha", "rgb", "ids", "roles"):
                        assert getattr(job.shot.deep, channel).is_pinned()
        serial = [row.to("cuda") for row in pinned]
        for start_step, workers in ((0, 0), (3, 2)):
            with device_batches(
                NativeBatches(
                    [frame], online, [[0]], {**cuda_config, "loader_workers": workers}, start_step
                ),
                "cuda",
            ) as overlap:
                for expected, actual in zip(serial[start_step:], overlap, strict=True):
                    a, b = expected.prepared, actual.prepared
                    assert a is not None
                    assert b is not None
                    assert (a.clean, a.live, a.selections) == (b.clean, b.live, b.selections)
                    torch.testing.assert_close(a.images, b.images, atol=0, rtol=0)
                    assert a.normalized is None or b.normalized is not None
                    if a.normalized is not None:
                        torch.testing.assert_close(a.normalized, b.normalized, atol=0, rtol=0)
                    for key, value in a.target.items():
                        torch.testing.assert_close(value, b.target[key], atol=0, rtol=0)
                    if a.cached_inputs is not None:
                        assert b.cached_inputs is not None
                        for key, value in a.cached_inputs.items():
                            torch.testing.assert_close(value, b.cached_inputs[key], atol=0, rtol=0)
                    scratch = torch.empty(1024 * 1024, device="cuda")
                    scratch.fill_(3)

    if loader_math == "torch":
        from strata_training.augment.pipeline import photometric, recompose
        from strata_training.data.targets import composite
        from strata_training.types import Layers

        augmented = OnlineLoader(
            [frame],
            samples,
            0,
            {
                "recompose": 0,
                "instance_grade": 1,
                "scene_grade": 1,
                "occluders": 1,
                "grain": 1,
                "camera": 1,
                "codec_weights": [0, 1, 0, 0],
            },
        )
        raw = NativeBatches(
            [frame], augmented, [[0]], {**config, "crop": 16, "halo": 8, "loader_math": "torch"}
        )[0]
        assert raw.prepared is not None
        expected_images: list[Array] = []
        expected_targets: dict[str, list[Array]] = {k: [] for k in raw.target}
        for job in raw.prepared.jobs:
            i, y, x = raw.selections[job.position]
            cpu_shot = augmented.shot(i).crop(y, x, 32, 32)
            layers = Layers(
                *(
                    frame.arrays[k][..., y : y + 32, x : x + 32]
                    for k in ("inverse_depth", "alpha", "rgb")
                )
            )
            rng = np.random.default_rng(job.seed)
            image, layers = recompose(cpu_shot, job.roles, layers, None, rng, job.config, {})
            expected_images.append(photometric(image, rng, job.config, {}))
            values = {
                **layers.arrays(),
                "beauty": composite(layers.alpha, layers.rgb),
                "mixed": (layers.alpha[0] > 0.03) & (layers.alpha[0] < 0.97),
            }
            for key, value in values.items():
                expected_targets[key].append(value[..., 8:24, 8:24])
        actual = raw.to(math_device).prepared
        replay = raw.to(math_device).prepared
        assert actual is not None
        assert replay is not None
        np.testing.assert_allclose(
            actual.images.cpu().numpy(), np.stack(expected_images), atol=5e-5, rtol=2e-5
        )
        torch.testing.assert_close(actual.images, replay.images, atol=0, rtol=0)
        for key, values in expected_targets.items():
            np.testing.assert_allclose(
                actual.target[key].cpu().numpy(), np.stack(values), atol=2e-6, rtol=2e-6
            )
            torch.testing.assert_close(actual.target[key], replay.target[key], atol=0, rtol=0)
        if math_device == "cuda":
            from strata_training.data.workers import device_batches

            augmented_config = {
                **config,
                "device": "cuda",
                "loader_math": "torch",
                "crop": 16,
                "halo": 8,
                "steps": 3,
            }
            references = [
                row.to("cuda")
                for row in batches(NativeBatches([frame], augmented, [[0]], augmented_config))
            ]
            consumed: list[LoadedBatch] = []
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            with device_batches(
                NativeBatches([frame], augmented, [[0]], augmented_config), "cuda"
            ) as overlap:
                for loaded in overlap:
                    consumed.append(loaded.map(lambda value: value.clone()))
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            for expected, observed in zip(references, consumed, strict=True):
                a, b = expected.prepared, observed.prepared
                assert a is not None
                assert b is not None
                torch.testing.assert_close(a.images, b.images, atol=0, rtol=0)
                torch.testing.assert_close(a.normalized, b.normalized, atol=0, rtol=0)
                for key, value in a.target.items():
                    torch.testing.assert_close(value, b.target[key], atol=0, rtol=0)
            with device_batches(
                NativeBatches([frame], augmented, [[0]], augmented_config), "cuda"
            ) as overlap:
                next(overlap)
            with pytest.raises(StopIteration):
                next(overlap)

    if math_device == "cuda" and loader_math == "torch":
        from strata_training.data.workers import device_batches

        damaged = np.load(samples / "sample/deep/z.npy")
        damaged[0] = np.nan
        save_array(samples / "sample/deep/z.npy", damaged)
        with (
            pytest.raises(ValueError, match="Raw samples"),
            device_batches(
                NativeBatches([frame], online, [[0]], {**config, "device": "cuda"}), "cuda"
            ) as overlap,
        ):
            list(overlap)

    assert sorted(p.name for p in (samples / "sample").iterdir()) == [
        "deep",
        "denoised.npy",
        "noisy.npy",
    ]
    write_json(root / "sample/context.json", {"image_sha256": "stale"})
    with pytest.raises(ValueError, match="match denoised"):
        OnlineLoader([frame], samples)
    write_json(manifest, {"instance_ids": [], "split": "test-only"})
    with pytest.raises(ValueError, match="test-only"):
        OnlineLoader([frame], samples)


def test_device_normalization_matches_clean_cache_color_transform(math_device: str) -> None:
    from strata_training.data.cache import display_rgb
    from strata_training.data.torch_online import normalize

    images = np.random.default_rng(719).uniform(-0.2, 3, (2, 3, 64, 96)).astype(np.float32)
    display = np.stack([display_rgb(image, "ACEScg") for image in images]).astype(np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    expected = (display - mean) / std
    actual = normalize(torch.tensor(images, device=math_device)).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_hybrid_policy_defaults_and_unsupported_temporal_or_oracle_inputs() -> None:
    config = training_defaults(
        {
            "mode": "head",
            "samples": "samples",
            "scale_mode": "raw",
            "learning_rate": 4e-4,
            "steps": 100,
            "batch_size": 2,
        }
    )
    assert (config["input_mode"], config["clean_fraction"], config["crop"], config["halo"]) == (
        "hybrid",
        0.4,
        512,
        0,
    )
    assert (
        config["early_stopping"],
        config["validation_every"],
        config["validation_patience"],
    ) == (
        True,
        250,
        4,
    )
    validate_config(config)
    assert config["loader_math"] == "torch"
    assert training_defaults({"device": "cpu"})["loader_math"] == "numpy"
    assert training_defaults({"input_mode": "cached"})["loader_math"] == "numpy"
    for key, value in (
        ("loader_math", "unknown"),
        ("loader_workers", -1),
        ("loader_workers", True),
        ("loader_prefetch", 0),
        ("loader_prefetch", 1.5),
    ):
        with pytest.raises(ValueError, match=key):
            validate_config({**config, key: value})
    for changes in ({"scale_mode": "oracle-aligned"}, {"clip_length": 2}, {"clean_fraction": -0.1}):
        with pytest.raises(ValueError, match=r"Hybrid native crops|clean_fraction"):
            validate_config({**config, **changes})
