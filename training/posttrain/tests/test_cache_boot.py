import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from strata_training.data import cache


@pytest.mark.parametrize("previous_boot", ["previous-boot", None])
def test_digest_rehashes_memo_from_another_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous_boot: str | None
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"new")
    stat = source.stat()
    fields = {name: getattr(stat, name) for name in ("st_size", "st_ino", "st_dev")}
    fields.update(st_mtime_ns=1, st_ctime_ns=1)
    record: dict[str, Any] = {
        "stat": [stat.st_size, 1, stat.st_ino, 1, stat.st_dev],
        "sha256": hashlib.sha256(b"old").hexdigest(),
        "hashed_ns": 10_000_000_000,
    }
    if previous_boot is not None:
        record["boot_id"] = previous_boot
    (tmp_path / "digests.json").write_text(json.dumps({str(source.resolve()): record}))
    original = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == Path("/proc/sys/kernel/random/boot_id"):
            return "current-boot\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    memo = cache.Digests(tmp_path)

    def file_stat(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(**fields)

    with monkeypatch.context() as metadata:
        metadata.setattr(Path, "stat", file_stat)
        assert memo.digest(source) == hashlib.sha256(b"new").hexdigest()
    memo.save()
    assert (
        json.loads((tmp_path / "digests.json").read_text())[str(source.resolve())]["boot_id"]
        == "current-boot"
    )
