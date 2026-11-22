import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from strata_training.data import cache


def test_restored_mtime_rebuilds_targets(
    packed_shot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "targets"
    hashed_ns = cache.time.time_ns() - 10_000_000_000
    monkeypatch.setattr(cache.time, "time_ns", lambda: hashed_ns)
    entries = cache.prepare(packed_shot.parent, output)
    source = packed_shot / "beauty/0001.exr"
    stat = source.stat()
    data = source.read_bytes()
    image = cache.read_rgb(source)

    def read_rgb(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return image

    monkeypatch.setattr(cache, "read_rgb", read_rgb)
    with source.open("r+b") as stream:
        stream.seek(len(data) - 1)
        stream.write(bytes([data[-1] ^ 1]))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source.stat().st_mtime_ns == stat.st_mtime_ns
    updated = cache.prepare(packed_shot.parent, output)
    assert updated[0]["cache"] != entries[0]["cache"]


@pytest.mark.parametrize("field", ["st_ctime_ns", "st_dev"])
def test_digest_memo_checks_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"old")
    stat = source.stat()
    fields = {name: getattr(stat, name) for name in ("st_size", "st_ino", "st_dev")}
    fields.update(st_mtime_ns=1, st_ctime_ns=1)

    def file_stat(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(**fields)

    memo = cache.Digests(tmp_path)
    monkeypatch.setattr(Path, "stat", file_stat)
    old = memo.digest(source)
    source.write_bytes(b"new")
    fields[field] += 1
    assert memo.digest(source) != old


@pytest.mark.parametrize("field", ["st_mtime_ns", "st_ctime_ns"])
def test_same_tick_digest_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"old")
    stat = source.stat()
    fields = {name: getattr(stat, name) for name in ("st_size", "st_ino", "st_dev")}
    fields.update(st_mtime_ns=1, st_ctime_ns=1)
    fields[field] = 10_000_000_000
    monkeypatch.setattr(time, "time_ns", lambda: 11_000_000_000)

    def file_stat(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(**fields)

    memo = cache.Digests(tmp_path)
    monkeypatch.setattr(Path, "stat", file_stat)
    old = memo.digest(source)
    source.write_bytes(b"new")
    assert memo.digest(source) != old
