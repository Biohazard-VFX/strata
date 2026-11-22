import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from strata_training.io import atomic_path, save_array, save_arrays, write_json
from strata_training.train import train


def test_training_logs_groups_and_refuses_incompatible_resume(tmp_path: Path) -> None:
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
