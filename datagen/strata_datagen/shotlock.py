"""One writer per shot: `<shot>/.lock` names its owner and exists while that owner changes the shot.

The lock file is created whole and exclusively (a hard link of a written temporary, which fails if
it exists). A lock is broken only when its owner is verifiably dead: same host, and another boot or
no live process with the recorded pid and start time. Breaking is serialized by `flock` on the shot
directory, so no two processes break the same lock.
"""

import fcntl
import json
import os
import secrets
import socket
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

LOCK = ".lock"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


class ShotLockedError(RuntimeError):
    """Another live process, or one that cannot be verified dead, holds the shot."""


def _boot() -> str:
    return BOOT_ID.read_text().strip()


def started(pid: int) -> str | None:
    """The start time of process `pid` in clock ticks since boot; None when no such process exists."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def owner() -> dict[str, object]:
    pid = os.getpid()
    return {
        "host": socket.gethostname(),
        "boot": _boot(),
        "pid": pid,
        "start": started(pid),
        "since": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def dead(record: dict[str, object]) -> bool:
    """True only when the owner is verifiably gone; an owner on another host never is."""
    if record["host"] != socket.gethostname():
        return False
    return record["boot"] != _boot() or started(cast("int", record["pid"])) != record["start"]


def _link(temporary: Path, lock: Path) -> bool:
    try:
        lock.hardlink_to(temporary)
    except FileExistsError:
        return False
    return True


def _record(lock: Path) -> dict[str, object] | None:
    try:
        return cast("dict[str, object]", json.loads(lock.read_text()))
    except FileNotFoundError:
        return None


@contextmanager
def _breaking(shot: Path) -> Generator[None]:
    descriptor = os.open(shot, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _refusal(shot: Path, record: dict[str, object]) -> ShotLockedError:
    return ShotLockedError(
        f"{shot} is locked by pid {record['pid']} on {record['host']} since {record['since']}; wait for it,"
        f" or remove {shot / LOCK} by hand once that process is known to be gone"
    )


def _acquire(shot: Path, temporary: Path) -> None:
    lock = shot / LOCK
    if _link(temporary, lock):
        return
    with _breaking(shot):
        record = _record(lock)
        if record is not None:
            if not dead(record):
                raise _refusal(shot, record)
            lock.unlink()
        if not _link(temporary, lock):
            raise _refusal(shot, cast("dict[str, object]", json.loads(lock.read_text())))


@contextmanager
def held(shot: Path) -> Generator[None]:
    """Hold the shot's writer lock for the block; raise ShotLockedError when another owner has it."""
    temporary = shot / f"{LOCK}.{os.getpid()}.{secrets.token_hex(4)}"
    temporary.write_text(json.dumps(owner(), sort_keys=True) + "\n")
    try:
        _acquire(shot, temporary)
    finally:
        temporary.unlink()
    try:
        yield
    finally:
        (shot / LOCK).unlink()
