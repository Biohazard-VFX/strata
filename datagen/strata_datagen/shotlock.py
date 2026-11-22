"""One writer per shot: an exclusive `flock` on `<shot>/.lock`, held while any process keeps its descriptor.

Liveness is the kernel's: the lock ends when the last descriptor of the locked open file closes, so a
renderer that inherits the descriptor keeps the shot locked after its parent dies, and no lock is ever
broken by reasoning about pids. A contender tries once without blocking and is refused. `.lock` is
never removed; `.lock.owner` names the latest holder for diagnostics only. Holds nest within a
process. The lock file lives inside the shot, so a shot moved while held stays locked at its new place,
and a contender whose lock path no longer leads to the file it locked refuses.
"""

import fcntl
import json
import os
import shlex
import socket
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from strata_datagen import files

LOCK = ".lock"
OWNER = ".lock.owner"


class ShotLockedError(RuntimeError):
    """Another open descriptor of the shot's lock file holds the lock."""


def owner() -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "since": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
    }


class Hold:
    """A held shot lock: `descriptor` keeps it; `location` follows the shot when `move` renames it."""

    def __init__(self, shot: Path, descriptor: int) -> None:
        self.location = shot
        self.descriptor = descriptor

    def move(self, destination: Path) -> None:
        """Rename the shot, lock file inside, to the absent `destination` on the same filesystem."""
        self.location.rename(destination)
        self.location = destination


_holds: dict[tuple[int, int], Hold] = {}


def _identity(descriptor: int) -> tuple[int, int]:
    status = os.fstat(descriptor)
    return status.st_dev, status.st_ino


def _path_identity(lock: Path) -> tuple[int, int] | None:
    try:
        status = lock.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return status.st_dev, status.st_ino


def holding(shot: Path) -> Hold | None:
    """This process's hold of `shot`, or None."""
    identity = _path_identity(shot / LOCK)
    return None if identity is None else _holds.get(identity)


def _refusal(shot: Path) -> ShotLockedError:
    try:
        record = json.loads((shot / OWNER).read_text())
        who = f"pid {record['pid']} on {record['host']} since {record['since']} ({record['command']})"
    except (FileNotFoundError, ValueError, KeyError):
        who = "another process"
    return ShotLockedError(f"{shot} is locked by {who}; wait for it to finish")


def _lock(shot: Path, descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise _refusal(shot) from None
    if _path_identity(shot / LOCK) != _identity(descriptor):
        raise ShotLockedError(f"{shot} was locked by a writer that moved it; look again")


def _acquire(shot: Path) -> Hold:
    descriptor = os.open(shot / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _lock(shot, descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return Hold(shot, descriptor)


@contextmanager
def held(shot: Path) -> Generator[Hold]:
    """Hold the shot's writer lock for the block; raise ShotLockedError when another holder has it."""
    outer = holding(shot)
    if outer is not None:
        yield outer
        return
    hold = _acquire(shot)
    identity = _identity(hold.descriptor)
    _holds[identity] = hold
    try:
        files.write_text(shot / OWNER, json.dumps(owner(), sort_keys=True) + "\n")
        yield hold
    finally:
        del _holds[identity]
        os.close(hold.descriptor)
