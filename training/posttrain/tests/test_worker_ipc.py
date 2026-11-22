import errno
import multiprocessing
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest

from strata_training.data.workers import LoadedBatch, NativeBatches, batches


class EmptyBatches(NativeBatches):
    def __getitem__(self, index: int) -> LoadedBatch:
        return LoadedBatch([], {}, {}, [], None)


def test_worker_socket_failure_names_tmpdir_before_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    process: Any = multiprocessing.current_process()
    monkeypatch.setitem(process._config, "tempdir", str(tmp_path))
    addresses: list[str] = []

    def denied(self: socket.socket, address: str) -> None:
        assert self.family == socket.AF_UNIX
        addresses.append(address)
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(socket.socket, "bind", denied)
    config = {"loader_workers": 1, "loader_prefetch": 2, "seed": 1, "device": "cpu"}
    with pytest.raises(RuntimeError, match="TMPDIR") as error:
        batches(NativeBatches([], None, [], config))
    assert str(tmp_path) in str(error.value)
    assert "Unix" in str(error.value)
    assert "Operation not permitted" in str(error.value)
    assert isinstance(error.value.__cause__, PermissionError)
    assert len(addresses) == 1
    assert Path(addresses[0]).is_relative_to(tmp_path)
    assert list(tmp_path.iterdir()) == []
    config["loader_workers"] = 0
    assert list(batches(NativeBatches([], None, [], {**config, "steps": 0}))) == []


@pytest.mark.parametrize("workers", [0, 1])
def test_loader_has_no_wall_clock_timeout(workers: int) -> None:
    config = {
        "loader_workers": workers,
        "loader_prefetch": 2,
        "seed": 1,
        "device": "cpu",
        "steps": 0,
    }
    loader = batches(NativeBatches([], None, [], config))
    assert loader.timeout == 0


def test_listener_rejects_76_character_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        prefix = f"/proc/self/fd/{descriptor}/"
        directory = Path(prefix + "x" * (76 - len(prefix)))
        directory.mkdir()
        assert len(str(directory)) == 76
        monkeypatch.setenv("TMPDIR", str(directory))
        monkeypatch.setattr(tempfile, "tempdir", str(directory))
        process: Any = multiprocessing.current_process()
        monkeypatch.delitem(process._config, "tempdir", raising=False)
        config = {"loader_workers": 1, "loader_prefetch": 2, "seed": 1, "device": "cpu"}
        with pytest.raises(RuntimeError, match="AF_UNIX path too long") as error:
            batches(NativeBatches([], None, [], config))
        assert str(directory) in str(error.value)
        assert "exFAT" not in str(error.value)
        assert isinstance(error.value.__cause__, OSError)
    finally:
        os.close(descriptor)


def test_spawned_worker_checks_its_own_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / ("x" * 76)
    directory.mkdir()
    monkeypatch.setenv("TMPDIR", str(directory))
    with tempfile.TemporaryDirectory() as parent:
        monkeypatch.setattr(tempfile, "tempdir", parent)
        process: Any = multiprocessing.current_process()
        monkeypatch.setitem(process._config, "tempdir", parent)
        config = {
            "loader_workers": 1,
            "loader_prefetch": 2,
            "seed": 1,
            "device": "cpu",
            "steps": 1,
        }
        loader = batches(EmptyBatches([], None, [], config))
        monkeypatch.setitem(process._config, "tempdir", str(directory))
        iterator: Any = iter(loader)
        try:
            with pytest.raises(RuntimeError, match="TMPDIR") as error:
                next(iterator)
            assert str(directory) in str(error.value)
            assert "AF_UNIX path too long" in str(error.value)
            assert "DataLoader worker process 0" in str(error.value)
        finally:
            iterator._shutdown_workers()
