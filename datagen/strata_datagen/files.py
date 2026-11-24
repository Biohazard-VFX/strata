"""Atomic file output: write a temporary sibling, then rename it over the target."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def replacing(path: Path) -> Generator[Path]:
    """Yield a temporary path with the target's suffix; it replaces `path` only if the block succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_durable(path: Path, text: str) -> None:
    with replacing(path) as temporary:
        temporary.write_text(text)
        sync(temporary)
    sync(path.parent)


def write_text(path: Path, text: str) -> None:
    with replacing(path) as temporary:
        temporary.write_text(text)
