import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from strata_training.data.cache import prune
from strata_training.io import write_json


@pytest.mark.parametrize("reader", ["train", "features", "evaluate", "predict", "baselines"])
@pytest.mark.parametrize("readonly", [False, True])
def test_readers_lock_referenced_caches_not_index_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, readonly: bool
) -> None:
    directories = [tmp_path / name for name in ("targets-b", "targets-a")]
    entries: list[dict[str, str]] = []
    for directory in directories:
        directory.mkdir()
        entry = {"id": directory.name, "cache": str(directory / "live.npz"), "shot": "shot"}
        entries.append(entry)
        write_json(directory / "index.json", [entry])
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    index = indexes / "index.json"
    write_json(index, [entries[0], entries[0]] if reader == "train" else entries + entries)
    validation = indexes / "validation.json"
    write_json(validation, [entries[1]])
    config = tmp_path / "config.json"
    write_json(config, {"index": str(index), "validation_index": str(validation)})
    calls: list[bool] = []

    def running(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        for directory in directories:
            blocked = subprocess.run(
                [sys.executable, "-m", "strata_training.cli", "cache-prune", str(directory)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert blocked.returncode != 0
            assert "in use" in blocked.stderr
        assert not (indexes / ".cache.lock").exists()
        calls.append(True)
        raise InterruptedError("reader reached")

    original_open = Path.open
    acquired: list[Path] = []

    def guarded_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if readonly and path.parent == indexes and any(flag in mode for flag in "wax+"):
            raise PermissionError("Index directory is read-only")
        if path.name == ".cache.lock":
            acquired.append(path.parent)
        return original_open(path, mode, *args, **kwargs)

    if readonly:
        indexes.chmod(0o555)
    monkeypatch.setattr(Path, "open", guarded_open)

    def invoke() -> None:
        if reader == "train":
            module = import_module("strata_training.train")
            monkeypatch.setattr(module, "_train", running)
            module.train(config)
        elif reader == "features":
            module = import_module("strata_training.model.features")
            monkeypatch.setattr(module, "_load", running)
            module.extract(index, tmp_path / "features", "depthpro2-s", device="cpu")
        elif reader == "predict":
            module = import_module("strata_training.eval.predict")
            monkeypatch.setattr(module, "load_model", running)
            module.cache_predictions(index, index, config, tmp_path / "predictions", device="cpu")
        elif reader == "baselines":
            module = import_module("strata_training.baselines.prepare")
            monkeypatch.setattr(module, "entries_for", running)
            module.prepare_baselines(index, index, tmp_path, tmp_path, config, config)
        else:
            module = import_module("strata_training.eval.runner")
            monkeypatch.setattr(module, "entries_for", running)
            module.evaluate(index, tmp_path / "protocol.json", tmp_path / "eval", [], device="cpu")

    try:
        with pytest.raises(InterruptedError, match="reader reached"):
            invoke()
    finally:
        indexes.chmod(0o755)
    assert calls == [True]
    assert acquired == sorted(directories)
    for directory in directories:
        assert prune(directory) == []
    assert json.loads(index.read_text())
