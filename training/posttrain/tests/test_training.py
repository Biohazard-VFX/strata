import hashlib
import json
import pickle
from importlib import import_module
from itertools import chain, repeat
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from strata_training.data.cache import CACHE_VERSION
from strata_training.data.feature_cache import base_arrays
from strata_training.io import atomic_path, save_array, save_arrays, write_json
from strata_training.train import train


def _configuration(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    cache, index, protocol = (tmp_path / name for name in ("frame.npz", "index.json", "split.json"))
    alpha = np.zeros((4, 32, 32), dtype=np.float32)
    inverse = alpha.copy()
    alpha[:2] = np.array([0.2, 1], dtype=np.float32)[:, None, None]
    inverse[:2] = np.array([0.5, 0.25], dtype=np.float32)[:, None, None]
    save_arrays(
        cache,
        version=np.array(CACHE_VERSION),
        alpha=alpha,
        inverse_depth=inverse,
        rgb=np.zeros((4, 3, 32, 32)),
        beauty=np.zeros((3, 32, 32)),
        mixed=np.ones((32, 32), dtype=bool),
    )
    root = tmp_path / "features"
    save_array(root / "train/features.npy", np.zeros((128, 32, 32), dtype=np.float32))
    for level in range(4):
        save_array(root / f"train/pyramid_{level}.npy", np.zeros((1, 1, 1)))
    save_arrays(root / "train/base.npz", **base_arrays(np.full((32, 32), 0.5), 1.0))
    write_json(
        index, [{"id": "train", "shot": "a", "frame": 1, "split": "train", "cache": str(cache)}]
    )
    write_json(
        protocol,
        {
            "train": ["train"],
            "version": 2,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
    )
    config = {
        "index": str(index),
        "protocol": str(protocol),
        "features": str(root),
        "preset": "depthpro2-h",
        "layers": 4,
        "mode": "head",
        "input_mode": "cached",
        "early_stopping": False,
        "seed": 4004,
        "crop": 32,
        "halo": 0,
        "batch_size": 1,
        "steps": 4,
        "learning_rate": 4e-4,
        "log_every": 2,
        "output": str(tmp_path / "run"),
        "device": "cpu",
    }
    path = tmp_path / "config.json"
    write_json(path, config)
    return config, path


def test_frame_pickle_and_crop_reads_are_bounded_and_exact(tmp_path: Path) -> None:
    from strata_training.data.dataset import Frame, batch

    config, _ = _configuration(tmp_path)
    entry = json.loads(Path(config["index"]).read_text())[0]
    frame = Frame(entry, Path(config["features"]))
    assert len(pickle.dumps(frame)) < 2048
    assert isinstance(frame.base, np.memmap)
    assert isinstance(frame.arrays["alpha"], np.memmap)
    with np.load(entry["cache"]) as archive:
        _, targets, _ = batch([frame], [(0, 0, 0)], 32, 0, "cpu")
        for key, value in targets.items():
            np.testing.assert_array_equal(value.numpy()[0], archive[key])
    restored = pickle.loads(pickle.dumps(frame))
    assert restored.crop(np.random.default_rng(3), 32) == frame.crop(np.random.default_rng(3), 32)
    save_arrays(
        Path(entry["cache"]), version=np.array(CACHE_VERSION), mixed=np.zeros((32, 32), dtype=bool)
    )
    with pytest.raises(ValueError, match="Source cache changed"):
        Frame(entry, Path(config["features"]))


def test_worker_batches_and_resumed_steps_are_identical(tmp_path: Path) -> None:
    from strata_training.config import training_defaults
    from strata_training.data.dataset import Frame
    from strata_training.data.workers import NativeBatches, batches

    config, _ = _configuration(tmp_path)
    config = training_defaults(config)
    entry = json.loads(Path(config["index"]).read_text())[0]
    frames = [Frame(entry, Path(config["features"]))]
    expected = list(batches(NativeBatches(frames, None, [[0]], config)))
    parallel = list(batches(NativeBatches(frames, None, [[0]], {**config, "loader_workers": 2})))
    resumed = list(batches(NativeBatches(frames, None, [[0]], config, 2)))
    for reference, actual in zip(expected, parallel, strict=True):
        assert reference.selections == actual.selections
        for key, value in reference.inputs.items():
            torch.testing.assert_close(value, actual.inputs[key], rtol=0, atol=0)
        for key, value in reference.target.items():
            torch.testing.assert_close(value, actual.target[key], rtol=0, atol=0)
    for reference, actual in zip(expected[2:], resumed, strict=True):
        assert reference.selections == actual.selections
        for key, value in reference.target.items():
            torch.testing.assert_close(value, actual.target[key], rtol=0, atol=0)


@pytest.mark.parametrize("workers", [0, 2])
def test_training_logs_groups_and_refuses_incompatible_resume(tmp_path: Path, workers: int) -> None:
    config, path = _configuration(tmp_path)
    config["loader_workers"] = workers
    write_json(path, config)
    report = train(path)
    assert report["train_frames"] == 1
    history = json.loads((tmp_path / "run/history.json").read_text())
    assert [row["step"] for row in history] == [2, 4]
    assert history[-1]["gradient_norms"]["head"] > 0
    assert set(history[-1]["last_terms"]) == {"alpha", "depth", "gradient", "rgb", "composite"}
    state = torch.load(tmp_path / "run/final.pt", weights_only=True)
    assert state["step"] == 4
    assert state["config"]["objective"] == "coverage-simplex-bce-v1"
    for key, value in (("steps", 5), ("learning_rate", 1e-3), ("warmup_steps", 1)):
        changed = {**config, "resume": str(tmp_path / "run/final.pt"), key: value}
        write_json(path, changed)
        with pytest.raises(ValueError, match="Resume model/data"):
            train(path)
    for key in ("input_mode", "clean_fraction", "compile_encoder"):
        state["config"].pop(key, None)
    historical = tmp_path / "historical.pt"
    with atomic_path(historical) as temporary:
        torch.save(state, temporary)
    write_json(path, {**config, "resume": str(historical)})
    with pytest.raises(ValueError, match="already reaches"):
        train(path)


def test_cached_training_refuses_protected_feature_roots(tmp_path: Path) -> None:
    config, path = _configuration(tmp_path)
    write_json(Path(config["features"]) / "manifest.json", {"split": "test-only"})
    with pytest.raises(ValueError, match="test-only"):
        train(path)


@pytest.mark.parametrize("rising", [False, True])
@pytest.mark.parametrize("baseline", [0.1, 0.39998, 1.0])
@pytest.mark.parametrize("passing", [False, True])
def test_ood_validation_retains_best_and_stops_at_patience_without_resetting_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing: bool,
    baseline: float,
    rising: bool,
) -> None:
    import shutil

    config, path = _configuration(tmp_path)
    entries = json.loads(Path(config["index"]).read_text())
    root = Path(config["features"])
    shutil.copytree(root / "train", root / "validation")
    cache = tmp_path / "validation.npz"
    shutil.copy2(entries[0]["cache"], cache)
    entry = {**entries[0], "id": "validation", "shot": "ood", "cache": str(cache)}
    index, protocol = tmp_path / "validation-index.json", tmp_path / "validation-protocol.json"
    write_json(index, [entry])
    write_json(
        protocol,
        {
            "test": [entry["id"]],
            "version": 2,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
    )
    config.update(
        early_stopping=True,
        validation_every=1 if passing else 2,
        validation_patience=2,
        steps=20,
        validation_index=str(index),
        validation_protocol=str(protocol),
        validation_features=str(root),
        log_every=1,
    )
    write_json(path, config)
    metrics = (
        chain(
            [
                (0.4, 0.3, 0.5),
                (0.3, 0.2, 0.4),
                (0.2, 0.0005, 0.39999),
                (0.1, 0.0009, 0.39998),
                (0.05, 0.001, 0.4 if rising else 0.39998),
            ],
            repeat((0.01, 0.001, 0.41 if rising else 0.39998)),
        )
        if passing
        else chain([(0.4, 0.3, 0.4), (0.3, 0.2, 0.3), (0.2, 0.4, 0.2)], repeat((0.1, 0.3, 0.3)))
    )

    def evaluate(model: object, config: object, frames: object) -> dict[str, Any]:
        del model, config, frames
        value, leak, sweep = next(metrics)
        return {
            "coverage_sweep": {"selection": sweep},
            "mixed_alpha_mae": value,
            "loss": value * 2,
            "mixed_pixels": 1024,
            "frames": 1,
            "leakage": {"worst_mean": leak, "passed": leak <= 0.001, "all": {"mean": leak}},
        }

    trainer: Any = import_module("strata_training.train")
    baseline_calls: list[object] = []

    def evaluate_baseline(frames: object) -> dict[str, Any]:
        baseline_calls.append(frames)
        return {"selection": baseline, "reference": "layer-targets"}

    monkeypatch.setattr(trainer, "evaluate_dp2_baseline", evaluate_baseline)
    monkeypatch.setattr(trainer, "evaluate_validation", evaluate)
    report = train(path)
    assert report["stopped_step"] == (5 if passing else 8)
    assert report["best_step"] == (3 if passing else 2)
    assert report["stopping_reason"] == "validation_patience"
    assert report["best_validation_mixed_alpha_mae"] == (0.1 if passing else 0.3)
    assert report["candidate_status"] == (
        "production candidate" if passing and baseline > 0.39998 else "no production candidate"
    )
    output = Path(config["output"])
    best = torch.load(output / "best.pt", weights_only=True)
    final = torch.load(output / "final.pt", weights_only=True)
    assert best["step"] == (3 if passing else 2)
    assert final["step"] == (5 if passing else 8)
    assert not torch.equal(best["model"]["output.weight"], final["model"]["output.weight"])
    state = json.loads((output / "validation.json").read_text())
    assert [row["step"] for row in state["history"]] == (
        [0, 1, 2, 3, 4, 5] if passing else [0, 2, 4, 6, 8]
    )
    assert [row["mixed_alpha_mae"] for row in state["history"]] == (
        [0.4, 0.3, 0.2, 0.1, 0.05, 0.01] if passing else [0.4, 0.3, 0.2, 0.1, 0.1]
    )
    assert final["validation"]["stale_evals"] == 2
    assert len(baseline_calls) == 1
    assert state["baseline"]["selection"] == baseline
    assert report["dp2_baseline"]["selection"] == baseline
    assert final["validation"]["best_sweep"] == (0.39998 if passing else 0.2)
    final["config"].pop("selection")
    old_selection = output / "old-selection.pt"
    with atomic_path(old_selection) as temporary:
        torch.save(final, temporary)
    write_json(path, {**config, "resume": str(old_selection)})
    with pytest.raises(ValueError, match="Resume model/data"):
        train(path)
    write_json(path, {**config, "resume": str(output / "last.pt")})
    with pytest.raises(ValueError, match="exhausted validation patience"):
        train(path)
    if passing:
        write_json(path, {**config, "resume": str(output / "best.pt")})
        resumed = train(path)
        assert resumed["stopped_step"] == 5
        assert resumed["best_step"] == 3
        assert resumed["candidate_status"] == report["candidate_status"]
        assert len(baseline_calls) == 1


def test_validation_brightness_uses_gt_not_denoised_model_input(tmp_path: Path) -> None:
    from torch import nn

    from strata_training.data.dataset import Frame
    from strata_training.model.head import DeepHead
    from strata_training.validation import evaluate_validation

    config, _ = _configuration(tmp_path)
    entry = json.loads(Path(config["index"]).read_text())[0]
    with np.load(entry["cache"]) as data:
        arrays = dict(data)
    arrays["mixed"][:, :16] = False
    arrays["beauty"][:] = 2
    save_arrays(Path(entry["cache"]), **arrays)
    image = np.zeros((3, 32, 32), dtype=np.float32)
    root = Path(config["features"]) / "train"
    save_array(root / "image.npy", image)
    write_json(root / "context.json", {"image_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    frame = Frame(entry, Path(config["features"]))
    assert frame.has_image
    assert not frame.arrays["beauty"].any()
    report = evaluate_validation(DeepHead(128, nn.Identity(), 4), config, [frame])
    assert report["leakage"]["bright"]["pixels"] == 384
    assert report["leakage"]["dark"]["pixels"] == 0


def test_validation_metric_is_pooled_native_mixed_alpha_and_preserves_model_mode(
    tmp_path: Path,
) -> None:
    from torch import nn

    from strata_training.data.dataset import Frame
    from strata_training.model.head import DeepHead
    from strata_training.validation import evaluate_validation, validation_frames

    config, _ = _configuration(tmp_path)
    entries = json.loads(Path(config["index"]).read_text())
    with np.load(entries[0]["cache"]) as data:
        arrays = dict(data)
    arrays["mixed"][:, :16] = False
    save_arrays(Path(entries[0]["cache"]), **arrays)
    frame = Frame(entries[0], Path(config["features"]))
    model = DeepHead(128, nn.Identity(), 4)
    result = evaluate_validation(model, config, [frame])
    assert result["mixed_alpha_mae"] == pytest.approx(0.05)
    assert result["mixed_pixels"] == 512
    assert result["leakage"]["all"]["pixels"] > 0
    assert result["leakage"]["all"]["mean"] == pytest.approx(0.75)
    assert result["loss"] > 0
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    all_mixed_cache = tmp_path / "all-mixed.npz"
    save_arrays(all_mixed_cache, **{**frame.arrays, "mixed": np.ones((32, 32), dtype=bool)})
    no_interior = Frame({**entries[0], "cache": str(all_mixed_cache)}, Path(config["features"]))
    with pytest.raises(ValueError, match="requires GT-opaque interior"):
        evaluate_validation(model, config, [no_interior])
    assert model.training
    empty_cache = tmp_path / "empty.npz"
    save_arrays(empty_cache, **{**frame.arrays, "mixed": np.zeros((32, 32), dtype=bool)})
    frame = Frame({**entries[0], "cache": str(empty_cache)}, Path(config["features"]))
    with pytest.raises(ValueError, match="requires mixed pixels"):
        evaluate_validation(model, config, [frame])
    assert model.training
    with pytest.raises(ValueError, match="explicit OOD validation"):
        validation_frames(config, entries)
    with pytest.raises(ValueError, match="separate scene-distribution index"):
        validation_frames(
            {
                **config,
                "validation_index": config["index"],
                "validation_protocol": config["protocol"],
                "validation_features": config["features"],
            },
            entries,
        )
    index, protocol = tmp_path / "alias-index.json", tmp_path / "alias-protocol.json"
    alias = {**entries[0], "id": "renamed", "shot": "different-label"}
    write_json(index, [alias])
    write_json(
        protocol,
        {
            "test": [alias["id"]],
            "version": 2,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="disjoint from training"):
        validation_frames(
            {
                **config,
                "validation_index": str(index),
                "validation_protocol": str(protocol),
                "validation_features": config["features"],
            },
            entries,
        )
    write_json(tmp_path / "manifest.json", {"split": "test-only"})
    with pytest.raises(ValueError, match="test-only"):
        evaluate_validation(model, config, [frame])


def test_dp2_baseline_is_single_opaque_layer_from_frozen_metric_depth(tmp_path: Path) -> None:
    from strata_training.data.dataset import Frame
    from strata_training.validation import evaluate_dp2_baseline

    config, _ = _configuration(tmp_path)
    entry = json.loads(Path(config["index"]).read_text())[0]
    with np.load(entry["cache"]) as archive:
        arrays = dict(archive)
    arrays["mixed"][:, :16] = False
    save_arrays(Path(entry["cache"]), **arrays)
    root = Path(config["features"]) / "train"
    save_arrays(root / "base.npz", **base_arrays(np.full((32, 32), 0.5), 2.0))
    frame = Frame(entry, Path(config["features"]))
    report = evaluate_dp2_baseline([frame])
    assert report["selection"] == pytest.approx(0.8 * np.log(2))
    assert report["edge"]["too_near"] == pytest.approx(0.8 * np.log(2))
    assert report["interior"]["too_near"] == pytest.approx(0.8 * np.log(2))
    assert report["edge"]["too_far"] == 0
