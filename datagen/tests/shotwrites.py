"""Record of every change inside a shot directory made without the writer lock.

An audit hook sees Python-level writes, including those relative to a directory descriptor; a
wrapper adds `os.open` with `dir_fd`, whose audit event omits the descriptor. Native writers (an
OpenEXR or USD library writing a file itself) raise no event, so a before/after snapshot of the
shots reports every change no observed event explains: native output must reach a shot by a
Python rename made under the lock.
"""

import os
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from strata_datagen import shotlock

SHOT_PARENTS = ("shots", "rejected")
MUTATING_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
TREE_EVENTS = ("os.rename", "os.rmdir")
PATH_ARGUMENTS: dict[str, tuple[tuple[int, int | None, bool], ...]] = {
    "os.rename": ((0, 2, False), (1, 3, True)),
    "os.link": ((1, 3, True),),
    "os.symlink": ((1, 2, True),),
    "os.remove": ((0, 1, False),),
    "os.rmdir": ((0, 1, False),),
    "os.mkdir": ((0, 2, True),),
    "os.truncate": ((0, None, False),),
}


class Watch:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.unlocked: list[str] = []
        self.unexplained: list[str] = []
        self.writes = 0
        self.touched: set[tuple[str, ...]] = set()
        self.trees: set[tuple[str, ...]] = set()


_active: list[Watch] = []


def _resolve(value: object, directory: object) -> Path | None:
    """The real path an event names; relative paths start at `directory` or the working directory."""
    if not isinstance(value, str | bytes | Path):
        return None
    path = Path(os.fsdecode(value))
    if not path.is_absolute():
        base = (
            Path(f"/proc/self/fd/{directory}").readlink()
            if isinstance(directory, int) and directory >= 0
            else "."
        )
        path = Path(base) / path
    return Path(os.path.realpath(path.parent)) / path.name


def _targets(event: str, args: tuple[object, ...]) -> list[tuple[object, object, bool]]:
    """(path, dir_fd, creates) for each path an event mutates; `creates` when it makes the path itself."""
    if event == "open":
        flags = args[2]
        return [(args[0], None, False)] if isinstance(flags, int) and flags & MUTATING_FLAGS else []
    return [(args[p], None if d is None else args[d], c) for p, d, c in PATH_ARGUMENTS.get(event, ())]


def _record(event: str, path: Path | None, *, creates: bool) -> None:
    watch = _active[-1]
    if path is None or not path.is_relative_to(watch.root):
        return
    parts = path.relative_to(watch.root).parts
    if len(parts) < 2 or parts[0] not in SHOT_PARENTS:
        return
    (watch.trees if event in TREE_EVENTS else watch.touched).add(parts)
    if (len(parts) == 2 and creates) or (len(parts) > 2 and parts[2].startswith(shotlock.LOCK)):
        return
    watch.writes += 1
    if shotlock.holding(watch.root / parts[0] / parts[1]) is None:
        watch.unlocked.append(f"{event} {path.relative_to(watch.root).as_posix()}")


def _hook(event: str, args: tuple[object, ...]) -> None:
    if not _active:
        return
    for value, directory, creates in _targets(event, args):
        _record(event, _resolve(value, directory), creates=creates)


sys.addaudithook(_hook)


def _open_at(original: Callable[..., int]) -> Callable[..., int]:
    def opened(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is not None and flags & MUTATING_FLAGS:
            _record("os.open", _resolve(path, dir_fd), creates=False)
        return original(path, flags, mode, dir_fd=dir_fd)

    return opened


def _snapshot(root: Path) -> dict[tuple[str, ...], tuple[object, ...]]:
    """Everything inside each shot but its lock files: kind and, for files, identity, size and time."""
    found: dict[tuple[str, ...], tuple[object, ...]] = {}
    for parent in SHOT_PARENTS:
        for folder, directories, names in os.walk(root / parent):
            for name in [*directories, *names]:
                path = Path(folder) / name
                parts = path.relative_to(root).parts
                if len(parts) < 3 or (len(parts) == 3 and name.startswith(shotlock.LOCK)):
                    continue
                status = path.lstat()
                kind = "link" if path.is_symlink() else "dir" if path.is_dir() else "file"
                found[parts] = (
                    (kind,) if kind == "dir" else (kind, status.st_ino, status.st_size, status.st_mtime_ns)
                )
    return found


def _explained(watch: Watch, parts: tuple[str, ...]) -> bool:
    return parts in watch.touched or any(parts[:n] in watch.trees for n in range(2, len(parts) + 1))


@contextmanager
def watching(root: Path) -> Generator[Watch]:
    watch = Watch(Path(os.path.realpath(root)))
    before = _snapshot(watch.root)
    _active.append(watch)
    try:
        with mock.patch.object(os, "open", _open_at(os.open)):
            yield watch
    finally:
        _active.pop()
        after = _snapshot(watch.root)
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        watch.unexplained = sorted("/".join(p) for p in changed if not _explained(watch, p))
