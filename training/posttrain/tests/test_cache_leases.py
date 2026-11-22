import os
from pathlib import Path
from typing import Any

import pytest

from strata_training.data import cache


@pytest.mark.parametrize("lock_exists", [False, True])
def test_read_only_cache_keeps_reader_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_exists: bool
) -> None:
    lock = tmp_path / ".cache.lock"
    if lock_exists:
        lock.touch()
    original_open = Path.open
    original_access = os.access

    def open_file(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path.parent == tmp_path and any(flag in mode for flag in "wa+"):
            raise PermissionError("Read-only cache")
        return original_open(path, mode, *args, **kwargs)

    def access(path: Any, mode: int, **kwargs: Any) -> bool:
        if path == tmp_path and mode == os.W_OK:
            return False
        return original_access(path, mode, **kwargs)

    with monkeypatch.context() as permissions:
        permissions.setattr(Path, "open", open_file)
        permissions.setattr(os, "access", access)
        if not lock_exists:
            with (
                pytest.raises(FileNotFoundError),
                cache.cache_lock(tmp_path, shared=True),
            ):
                pytest.fail("Readers need an existing lease file")
            return
        with cache.cache_lock(tmp_path, shared=True), cache.cache_lock(tmp_path, shared=True):
            with pytest.raises(PermissionError):
                cache.prune(tmp_path)
            with monkeypatch.context() as writer:
                writer.setattr(Path, "open", original_open)
                writer.setattr(os, "access", original_access)
                with (
                    pytest.raises(ValueError, match="in use"),
                    cache.cache_lock(tmp_path, shared=False),
                ):
                    pytest.fail("A read-only reader must block writable maintenance")


@pytest.mark.parametrize("shared", [False, True])
def test_cache_lock_reports_blocking_owner(tmp_path: Path, shared: bool) -> None:
    message = "writer or prune" if shared else "stop training"
    with (
        cache.cache_lock(tmp_path, shared=not shared),
        pytest.raises(ValueError, match=message),
        cache.cache_lock(tmp_path, shared=shared),
    ):
        pytest.fail("Conflicting lease was granted")
