from io import BufferedReader
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_review_regressions import prepared_frames

from strata_training.data.online import OnlineLoader
from strata_training.data.targets import composite
from strata_training.data.torch_online import finalize, prepare_raw


@pytest.mark.parametrize("clean_fraction", [0.0, 1.0])
def test_every_view_uses_final_layer_composite(
    packed_shot: Path, tmp_path: Path, math_device: str, clean_fraction: float
) -> None:
    frames, samples = prepared_frames(packed_shot, tmp_path)
    from strata_training.data.dataset import Frame
    from strata_training.io import save_arrays

    for frame in frames:
        cache = Path(frame.entry["cache"]).with_name(frame.entry["id"] + ".rounded.npz")
        with np.load(frame.entry["cache"]) as archive:
            values = {k: archive[k] for k in archive.files}
        values["target_beauty"] = values["beauty"].astype(np.float32) + np.float32(2**-14)
        save_arrays(cache, **values)
        frame.entry = {**frame.entry, "cache": str(cache)}
    frames = [Frame(frame.entry, frame.feature_root) for frame in frames]
    loader = OnlineLoader(
        frames,
        samples,
        clean_fraction,
        dict.fromkeys(
            ("scene_grade", "instance_grade", "recompose", "occluders", "grain", "camera"), 0
        ),
    )
    selections = [(0, 0, 0), (1, 0, 0)]
    numpy = loader.prepare(selections, np.random.default_rng(71), 1, 0, "cpu")
    raw = prepare_raw(loader, selections, np.random.default_rng(71), 1, 0)
    raw.images = raw.images.to(math_device)
    raw.target = {k: v.to(math_device) for k, v in raw.target.items()}
    raw.jobs = [job.map(lambda value: value.to(math_device)) for job in raw.jobs]
    tensor = finalize(raw, 1, 0)
    from strata_training.data.dataset import batch as cached_batch

    _, cached, _ = cached_batch(frames, selections, 1, 0, "cpu")
    for targets in (numpy.target, tensor.target, cached):
        for position in range(2):
            expected = composite(
                targets["alpha"][position].cpu().numpy(),
                targets["rgb"][position].cpu().numpy(),
            )
            np.testing.assert_array_equal(targets["beauty"][position].cpu().numpy(), expected)
    for key, value in numpy.target.items():
        torch.testing.assert_close(value, tensor.target[key].cpu(), atol=0, rtol=0)


def test_removed_parameters_are_absent() -> None:
    import inspect

    from strata_training.augment.pipeline import augment_sample, recompose
    from strata_training.augment.torch_compose import recompose as tensor_recompose
    from strata_training.augment.torch_compose import recompose_samples
    from strata_training.data.scale import oracle_metric_scale

    for function in (
        augment_sample,
        recompose,
        tensor_recompose,
        recompose_samples,
    ):
        assert "shot_roles" not in inspect.signature(function).parameters
    assert "entry" not in inspect.signature(oracle_metric_scale).parameters


def test_warm_prepare_memoizes_exr_digests(
    packed_shot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strata_training.data import cache

    output = tmp_path / "targets"
    original = cache.hashlib.file_digest
    calls: list[str] = []

    def counted(source: BufferedReader, digest: str) -> Any:
        calls.append(str(source.name))
        return original(source, digest)

    monkeypatch.setattr(cache.hashlib, "file_digest", counted)
    entries = cache.prepare(packed_shot.parent, output)
    assert len(calls) == 4
    calls.clear()
    assert cache.prepare(packed_shot.parent, output) == entries
    assert calls == []
    source = packed_shot / "beauty/0001.exr"
    import os

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert cache.prepare(packed_shot.parent, output) == entries
    assert calls == [str(source)]
    calls.clear()
    replacement = source.with_suffix(".new.exr")
    replacement.write_bytes(source.read_bytes())
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    replacement.replace(source)
    assert cache.prepare(packed_shot.parent, output) == entries
    assert calls == [str(source)]
    calls.clear()
    source.write_bytes(source.read_bytes() + b"\x00")
    updated = cache.prepare(packed_shot.parent, output)
    assert updated[0]["cache"] != entries[0]["cache"]
    assert calls == [str(source)]


def test_offline_prune_removes_only_unreferenced_cache_files(
    packed_shot: Path, tmp_path: Path
) -> None:
    from strata_training.data import cache

    output = tmp_path / "targets"
    entries = cache.prepare(packed_shot.parent, output)
    orphan = output / "obsolete.deadbeef.npz"
    np.savez(orphan, alpha=np.ones(1))
    from strata_training.data.mapped import unpack

    mapping = unpack(orphan)
    retained = unpack(Path(entries[0]["cache"]))
    unrelated = output / "keep.json"
    unrelated.write_text("{}")
    with cache.cache_lock(output, shared=True), pytest.raises(ValueError, match="in use"):
        cache.prune(output)
    assert orphan.exists()
    assert mapping.exists()
    assert cache.prune(output) == [orphan, mapping]
    assert not orphan.exists()
    assert not mapping.exists()
    assert unrelated.exists()
    assert retained.exists()
    assert all(Path(entry["cache"]).exists() for entry in entries)
    assert cache.prune(output) == []


def test_train_holds_target_cache_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from strata_training import train
    from strata_training.data import cache
    from strata_training.io import write_json

    output = tmp_path / "targets"
    output.mkdir()
    write_json(output / "index.json", [])
    config = tmp_path / "config.json"
    write_json(config, {"index": str(output / "index.json")})
    calls: list[bool] = []

    def running(path: Path) -> dict[str, bool]:
        assert path == config
        with pytest.raises(ValueError, match="in use"):
            cache.prune(output)
        calls.append(True)
        return {"done": True}

    monkeypatch.setattr(train, "_train", running, raising=False)
    assert train.train(config) == {"done": True}
    assert calls == [True]
    assert cache.prune(output) == []


def test_loader_reports_eligibility_and_configurable_warning(
    packed_shot: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    frames, samples = prepared_frames(packed_shot, tmp_path)
    with caplog.at_level(logging.INFO):
        loader = OnlineLoader(frames, samples, eligibility_warning_fraction=0.75)
    assert loader.eligible == frozenset({0})
    assert "1/2" in caplog.text
    assert "50.0%" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        OnlineLoader(frames, samples, eligibility_warning_fraction=0.25)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    for threshold in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="eligibility_warning_fraction"):
            OnlineLoader(frames, samples, eligibility_warning_fraction=threshold)



def test_hero_group_marker_is_named_once() -> None:
    import ast
    import inspect

    from strata_training.augment import pipeline, torch_compose

    markers: list[str] = []
    for module in (pipeline, torch_compose):
        tree = ast.parse(inspect.getsource(module))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and node.operand.value == 2
            ):
                parent = parents[node]
                if isinstance(parent, ast.Tuple):
                    continue
                assert isinstance(parent, ast.Assign)
                assert [target.id for target in parent.targets if isinstance(target, ast.Name)] == [
                    "HERO_GROUP"
                ]
                markers.append(module.__name__)
    assert markers == [pipeline.__name__]


def test_cache_prune_cli_obeys_cross_process_training_lock(
    packed_shot: Path, tmp_path: Path
) -> None:
    import subprocess
    import sys

    from strata_training.data.cache import cache_lock, prepare

    output = tmp_path / "targets"
    entries = prepare(packed_shot.parent, output)
    orphan = output / "unused.npz"
    np.savez(orphan, value=1)
    command = [sys.executable, "-m", "strata_training.cli", "cache-prune", str(output)]
    with cache_lock(output, shared=True):
        blocked = subprocess.run(command, capture_output=True, text=True, check=False)
    assert blocked.returncode != 0
    assert "in use" in blocked.stderr
    assert orphan.exists()
    success = subprocess.run(command, capture_output=True, text=True, check=True)
    import json

    assert json.loads(success.stdout) == {"removed": [str(orphan)]}
    assert not orphan.exists()
    assert all(Path(entry["cache"]).exists() for entry in entries)


def test_shared_composite_matches_independent_layer_weights(math_device: str) -> None:
    rng = np.random.default_rng(4891)
    alpha = rng.uniform(0, 1, (4, 8, 12)).astype(np.float32)
    rgb = rng.uniform(-1, 4, (4, 3, 8, 12)).astype(np.float32) * alpha[:, None]
    transmission = np.concatenate((np.ones_like(alpha[:1]), np.cumprod(1 - alpha[:-1], axis=0)))
    expected = (transmission[:, None] * rgb).sum(axis=0)
    np.testing.assert_array_equal(composite(alpha, rgb), expected)
    actual = composite(
        torch.tensor(alpha, device=math_device), torch.tensor(rgb, device=math_device)
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
