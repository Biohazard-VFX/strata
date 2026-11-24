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


def _ood_configuration(tmp_path: Path) -> tuple[dict[str, Any], Path]:
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
        validation_index=str(index),
        validation_protocol=str(protocol),
        validation_features=str(root),
    )
    return config, path


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
    config, path = _ood_configuration(tmp_path)
    config.update(
        early_stopping=True,
        validation_every=1 if passing else 2,
        validation_patience=2,
        steps=20,
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
        del model, config
        assert frames == baseline_calls[0]
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
    assert report["stopped_step"] == (5 if passing else 6)
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
    assert final["step"] == (5 if passing else 6)
    assert not torch.equal(best["model"]["output.weight"], final["model"]["output.weight"])
    state = json.loads((output / "validation.json").read_text())
    assert [row["step"] for row in state["history"]] == (
        [0, 1, 2, 3, 4, 5] if passing else [0, 2, 4, 6]
    )
    assert [row["mixed_alpha_mae"] for row in state["history"]] == (
        [0.4, 0.3, 0.2, 0.1, 0.05, 0.01] if passing else [0.4, 0.3, 0.2, 0.1]
    )
    assert final["validation"]["stale_evals"] == 2
    assert len(baseline_calls) == 1
    assert state["baseline"]["selection"] == baseline
    assert report["dp2_baseline"]["selection"] == baseline
    assert final["validation"]["best_sweep"] == (0.39998 if passing else None)
    final["config"].pop("selection", None)
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


@pytest.mark.parametrize(
    ("stream", "stopped_step", "best_step", "resume_step", "best_sweep", "lowest_leakage"),
    [
        (
            [(0.01, 0.1), (0.7, 0.0009), (0.001, 0.1), (0.6, 0.0008), (0.5, 0.0007)],
            6,
            4,
            2,
            0.5,
            0.1,
        ),
        ([(0.4, 0.0005), (0.01, 0.1), (0.001, 0.1)], 2, 0, 1, 0.4, None),
        (
            [(0.01, 0.5), (0.2, 0.4), (0.3, 0.3), (0.4, 0.2), (0.001, 0.2)],
            5,
            4,
            4,
            None,
            0.2,
        ),
    ],
)
def test_patience_tracks_eligible_sweep_or_prepass_leakage_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: list[tuple[float, float]],
    stopped_step: int,
    best_step: int,
    resume_step: int,
    best_sweep: float | None,
    lowest_leakage: float | None,
) -> None:
    import shutil

    config, path = _ood_configuration(tmp_path)
    config.update(early_stopping=True, validation_every=1, validation_patience=2, steps=12)
    write_json(path, config)
    metrics = chain(stream, repeat(stream[-1]))
    trainer: Any = import_module("strata_training.train")
    save_checkpoint = trainer.save_checkpoint
    resume = tmp_path / "resume.pt"

    def save(target: Path, *args: Any, **kwargs: Any) -> None:
        save_checkpoint(target, *args, **kwargs)
        if target.name == "last.pt" and args[4] == resume_step:
            shutil.copyfile(target, resume)

    def evaluate(model: object, config: object, frames: object) -> dict[str, Any]:
        del model, config, frames
        sweep, leak = next(metrics)
        return {
            "coverage_sweep": {"selection": sweep},
            "mixed_alpha_mae": 0.1,
            "loss": 0.2,
            "leakage": {"worst_mean": leak, "passed": leak <= 0.001, "all": {"mean": leak}},
        }

    def evaluate_baseline(frames: object) -> dict[str, Any]:
        del frames
        return {"selection": 0.6}

    monkeypatch.setattr(trainer, "evaluate_validation", evaluate)
    monkeypatch.setattr(trainer, "evaluate_dp2_baseline", evaluate_baseline)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    report = train(path)
    output = Path(config["output"])
    final = torch.load(output / "final.pt", weights_only=True)
    assert report["stopped_step"] == stopped_step
    assert report["best_step"] == best_step
    assert torch.load(output / "best.pt", weights_only=True)["step"] == best_step
    assert report["candidate_status"] == (
        "production candidate"
        if best_sweep is not None and best_sweep < 0.6
        else "no production candidate"
    )
    saved = torch.load(resume, weights_only=True)
    assert saved["step"] == resume_step
    assert saved["validation"]["stale_evals"] == 1
    assert final["validation"]["best_sweep"] == best_sweep
    assert final["validation"]["lowest_leakage"] == lowest_leakage
    assert final["validation"]["stale_evals"] == 2
    metrics = chain(stream[resume_step + 1 :], repeat(stream[-1]))
    write_json(path, {**config, "resume": str(resume)})
    resumed = train(path)
    restored = torch.load(output / "final.pt", weights_only=True)
    assert resumed["stopped_step"] == stopped_step
    assert resumed["best_step"] == best_step
    assert resumed["candidate_status"] == report["candidate_status"]
    for key in ("best_sweep", "lowest_leakage", "stale_evals", "best_metrics"):
        assert restored["validation"][key] == final["validation"][key]
    for key, value in final["model"].items():
        torch.testing.assert_close(restored["model"][key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("stream", "best_step"),
    [
        ([(0.6, 0.01), (0.3, 0.0005), (0.2, 0.01)], 1),
        ([(0.6, 0.0005), (0.3, 0.0005), (0.7, 0.0005)], 1),
        ([(0.6, 0.01), (0.5, 0.01), (0.3, 0.0005)], 2),
    ],
)
def test_candidate_status_belongs_to_kept_checkpoint_at_final_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: list[tuple[float, float]],
    best_step: int,
) -> None:
    config, path = _ood_configuration(tmp_path)
    config.update(early_stopping=True, validation_every=1, validation_patience=3, steps=2)
    write_json(path, config)
    metrics = iter(stream)

    def evaluate(model: object, config: object, frames: object) -> dict[str, Any]:
        del model, config, frames
        sweep, leak = next(metrics)
        return {
            "coverage_sweep": {"selection": sweep},
            "mixed_alpha_mae": 0.1,
            "loss": 0.2,
            "mixed_pixels": 1024,
            "frames": 1,
            "leakage": {"worst_mean": leak, "passed": leak <= 0.001, "all": {"mean": leak}},
        }

    def evaluate_baseline(frames: object) -> dict[str, Any]:
        del frames
        return {"selection": 0.4}

    trainer: Any = import_module("strata_training.train")
    monkeypatch.setattr(trainer, "evaluate_validation", evaluate)
    monkeypatch.setattr(trainer, "evaluate_dp2_baseline", evaluate_baseline)
    report = train(path)
    output = Path(config["output"])
    best = torch.load(output / "best.pt", weights_only=True)
    final = torch.load(output / "final.pt", weights_only=True)
    state = json.loads((output / "validation.json").read_text())
    assert report["stopped_step"] == final["step"] == 2
    assert report["best_step"] == best["step"] == best_step
    assert state["best_metrics"]["checkpoint_step"] == best_step
    assert [row["coverage_sweep"]["selection"] for row in state["history"]] == [
        sweep for sweep, _ in stream
    ]
    for result in (report, state, best["validation"], final["validation"]):
        assert result["candidate_status"] == "production candidate"


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


@pytest.mark.parametrize(("inverse", "near", "far"), [(0.5, 0.8, 0.0), (0.125, 0.0, 1.2)])
def test_dp2_baseline_is_single_opaque_layer_from_frozen_metric_depth(
    tmp_path: Path, inverse: float, near: float, far: float
) -> None:
    from strata_training.data.dataset import Frame
    from strata_training.validation import evaluate_dp2_baseline

    config, _ = _configuration(tmp_path)
    entry = json.loads(Path(config["index"]).read_text())[0]
    with np.load(entry["cache"]) as archive:
        arrays = dict(archive)
    arrays["mixed"][:, :16] = False
    save_arrays(Path(entry["cache"]), **arrays)
    root = Path(config["features"]) / "train"
    save_arrays(root / "base.npz", **base_arrays(np.full((32, 32), inverse), 2.0))
    first = Frame(entry, Path(config["features"]))
    second_root = Path(config["features"]) / "second"
    save_array(second_root / "features.npy", np.zeros((128, 32, 32), dtype=np.float32))
    second_inverse = 0.125 if inverse == 0.5 else 0.5
    second_near, second_far = (0.0, 1.2) if inverse == 0.5 else (0.8, 0.0)
    save_arrays(second_root / "base.npz", **base_arrays(np.full((32, 32), second_inverse), 2.0))
    arrays["mixed"][:] = False
    arrays["mixed"][:, :4] = True
    second_cache = tmp_path / "second.npz"
    save_arrays(second_cache, **arrays)
    second = Frame({**entry, "id": "second", "cache": str(second_cache)}, Path(config["features"]))
    report = evaluate_dp2_baseline([first, second])
    expected: list[float] = []
    for region, first_pixels, second_pixels in (("edge", 640, 256), ("interior", 384, 768)):
        pixels = first_pixels + second_pixels
        assert report[region]["pixels"] == pixels
        pooled_near = (near * first_pixels + second_near * second_pixels) / pixels
        pooled_far = (far * first_pixels + second_far * second_pixels) / pixels
        assert report[region]["too_near"] == pytest.approx(pooled_near * np.log(2))
        assert report[region]["too_far"] == pytest.approx(pooled_far * np.log(2))
        expected.append((pooled_near + pooled_far) * np.log(2))
    assert report["selection"] == pytest.approx(sum(expected) / 2)
    for mixed in (False, True):
        arrays["mixed"][:] = mixed
        missing_cache = tmp_path / f"missing-region-{mixed}.npz"
        save_arrays(missing_cache, **arrays)
        missing = Frame({**entry, "cache": str(missing_cache)}, Path(config["features"]))
        with pytest.raises(ValueError, match="requires finite edge and opaque-interior sweeps"):
            evaluate_dp2_baseline([missing])
