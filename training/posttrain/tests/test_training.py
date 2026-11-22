import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

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
    save_arrays(
        root / "train/base.npz", raw_inverse_depth=np.full((32, 32), 0.5), raw_metric_scale=1.0
    )
    write_json(
        index, [{"id": "train", "shot": "a", "frame": 1, "split": "train", "cache": str(cache)}]
    )
    write_json(
        protocol,
        {"train": ["train"], "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()},
    )
    config = {
        "index": str(index),
        "protocol": str(protocol),
        "features": str(root),
        "preset": "depthpro2-h",
        "layers": 4,
        "mode": "head",
        "scale_mode": "raw",
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


def test_training_logs_groups_and_refuses_incompatible_resume(tmp_path: Path) -> None:
    config, path = _configuration(tmp_path)
    report = train(path)
    assert report["train_frames"] == 1
    history = json.loads((tmp_path / "run/history.json").read_text())
    assert [row["step"] for row in history] == [2, 4]
    assert history[-1]["gradient_norms"]["head"] > 0
    assert set(history[-1]["last_terms"]) == {"alpha", "depth", "gradient", "rgb", "composite"}
    state = torch.load(tmp_path / "run/final.pt", weights_only=True)
    assert state["step"] == 4
    assert state["config"]["objective"] == "coverage-bce-logits-v1"
    for key, value in (("steps", 5), ("learning_rate", 1e-3), ("warmup_steps", 1)):
        changed = {**config, "resume": str(tmp_path / "run/final.pt"), key: value}
        write_json(path, changed)
        with pytest.raises(ValueError, match="Resume model/data/scale"):
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


def test_ood_validation_retains_best_and_stops_at_patience_without_resetting_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        {"test": [entry["id"]], "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()},
    )
    config.update(
        early_stopping=True,
        validation_every=2,
        validation_patience=2,
        steps=20,
        validation_index=str(index),
        validation_protocol=str(protocol),
        validation_features=str(root),
        log_every=1,
    )
    write_json(path, config)
    metrics = iter([0.4, 0.3, 0.31, 0.32])

    def evaluate(model: object, config: object, frames: object) -> dict[str, Any]:
        del model, config, frames
        value = next(metrics)
        return {"mixed_alpha_mae": value, "loss": value * 2, "mixed_pixels": 1024, "frames": 1}

    trainer: Any = import_module("strata_training.train")
    monkeypatch.setattr(trainer, "evaluate_validation", evaluate)
    report = train(path)
    assert report["stopped_step"] == 6
    assert report["best_step"] == 2
    assert report["stopping_reason"] == "validation_patience"
    assert report["best_validation_mixed_alpha_mae"] == 0.3
    output = Path(config["output"])
    best = torch.load(output / "best.pt", weights_only=True)
    final = torch.load(output / "final.pt", weights_only=True)
    assert best["step"] == 2
    assert final["step"] == 6
    assert not torch.equal(best["model"]["output.weight"], final["model"]["output.weight"])
    state = json.loads((output / "validation.json").read_text())
    assert [row["step"] for row in state["history"]] == [0, 2, 4, 6]
    assert [row["mixed_alpha_mae"] for row in state["history"]] == [0.4, 0.3, 0.31, 0.32]
    assert final["validation"]["stale_evals"] == 2
    write_json(path, {**config, "resume": str(output / "last.pt")})
    with pytest.raises(ValueError, match="exhausted validation patience"):
        train(path)


def test_validation_metric_is_pooled_native_mixed_alpha_and_preserves_model_mode(
    tmp_path: Path,
) -> None:
    from torch import nn

    from strata_training.data.dataset import Frame
    from strata_training.model.head import DeepHead
    from strata_training.validation import evaluate_validation, validation_frames

    config, _ = _configuration(tmp_path)
    entries = json.loads(Path(config["index"]).read_text())
    frame = Frame(entries[0], Path(config["features"]))
    model = DeepHead(128, nn.Identity(), 4)
    result = evaluate_validation(model, config, [frame])
    assert result["mixed_alpha_mae"] == pytest.approx(0.3)
    assert result["mixed_pixels"] == 1024
    assert result["loss"] > 0
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
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
        {"test": [alias["id"]], "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()},
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
