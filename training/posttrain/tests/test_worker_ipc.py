import multiprocessing
import queue
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest

from strata_training.data.workers import NativeBatches, batches


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
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(socket.socket, "bind", denied)
    config = {"loader_workers": 1, "loader_prefetch": 2, "seed": 1, "device": "cpu"}
    with pytest.raises(RuntimeError, match="TMPDIR") as error:
        batches(NativeBatches([], None, [], config))
    assert str(tmp_path) in str(error.value)
    assert "Unix" in str(error.value)
    assert isinstance(error.value.__cause__, PermissionError)
    assert len(addresses) == 1
    assert Path(addresses[0]).is_relative_to(tmp_path)
    assert list(tmp_path.iterdir()) == []
    config["loader_workers"] = 0
    assert list(batches(NativeBatches([], None, [], {**config, "steps": 0}))) == []


def test_worker_queue_failure_has_a_finite_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "loader_workers": 1,
        "loader_prefetch": 2,
        "seed": 1,
        "device": "cpu",
        "steps": 0,
    }
    loader = batches(NativeBatches([], None, [], config))
    iterator: Any = iter(loader)
    waits: list[float] = []

    def empty(block: bool = True, timeout: float | None = None) -> Any:
        assert block
        if waits or timeout is None or timeout <= 0:
            raise RuntimeError("unbounded worker queue wait")
        waits.append(timeout)
        raise queue.Empty

    monkeypatch.setattr(iterator._data_queue, "get", empty)
    try:
        with pytest.raises(RuntimeError, match="DataLoader timed out"):
            iterator._get_data()
        assert len(waits) == 1
        assert waits[0] <= 60
    finally:
        iterator._shutdown_workers()
