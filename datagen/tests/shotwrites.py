"""Audit-hook record of every write inside a shot directory made without the writer lock."""

import json
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from strata_datagen import shotlock

SHOT_PARENTS = ("shots", "rejected")
MUTATING_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class Watch:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.unlocked: list[str] = []
        self.writes = 0


_active: list[Watch] = []


def _path(value: object) -> Path | None:
    return Path(os.fsdecode(value)) if isinstance(value, str | bytes | Path) else None


def _held(shot: Path) -> bool:
    try:
        record = json.loads((shot / shotlock.LOCK).read_text())
    except (FileNotFoundError, NotADirectoryError):
        return False
    return record["pid"] == os.getpid()


def _targets(event: str, args: tuple[object, ...]) -> list[tuple[object, bool]]:
    """Paths an event mutates, each flagged when it is the creation of the path itself."""
    if event == "open":
        flags = args[2]
        return [(args[0], False)] if isinstance(flags, int) and flags & MUTATING_FLAGS else []
    if event == "os.rename":
        return [(args[0], False), (args[1], True)]
    if event in ("os.remove", "os.rmdir"):
        return [(args[0], False)]
    if event == "os.mkdir":
        return [(args[0], True)]
    if event == "os.link":
        return [(args[1], True)]
    return []


def _hook(event: str, args: tuple[object, ...]) -> None:
    if not _active:
        return
    watch = _active[-1]
    for value, creates in _targets(event, args):
        path = _path(value)
        if path is None or not path.is_relative_to(watch.root):
            continue
        parts = path.relative_to(watch.root).parts
        if len(parts) < 2 or parts[0] not in SHOT_PARENTS or (len(parts) == 2 and creates):
            continue
        if len(parts) > 2 and parts[2].startswith(shotlock.LOCK):
            continue
        watch.writes += 1
        if not _held(watch.root / parts[0] / parts[1]):
            watch.unlocked.append(f"{event} {path.relative_to(watch.root).as_posix()}")


sys.addaudithook(_hook)


@contextmanager
def watching(root: Path) -> Generator[Watch]:
    watch = Watch(root)
    _active.append(watch)
    try:
        yield watch
    finally:
        _active.pop()
