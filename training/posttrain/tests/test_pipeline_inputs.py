from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from strata_training.io import write_json


def test_default_pipeline_builds_features_from_the_supplied_denoised_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: Any = import_module("strata_training.pipeline")
    observed: dict[str, Path | None] = {}

    def ignore(*args: object, **kwargs: object) -> None:
        del args, kwargs

    def extract(
        index: Path,
        output: Path,
        preset: str,
        clip_length: int,
        device: str,
        samples: Path | None = None,
    ) -> None:
        del index, output, preset, clip_length, device
        observed["samples"] = samples

    for name in ("prepare", "freeze", "train", "evaluate"):
        monkeypatch.setattr(stages, name, ignore)
    monkeypatch.setattr(stages, "extract", extract)
    config = {
        "mode": "head",
        "preset": "depthpro2-s",
        "samples": str(tmp_path / "samples"),
        "dataset": "dataset",
        "index": "index.json",
        "protocol": "protocol.json",
        "features": "features",
        "output": "run",
        "layers": 4,
        "crop": 512,
        "halo": 0,
        "steps": 2,
        "batch_size": 1,
        "learning_rate": 4e-4,
    }
    path = tmp_path / "config.json"
    write_json(path, config)
    stages.pipeline(path)
    assert observed["samples"] == tmp_path / "samples"
