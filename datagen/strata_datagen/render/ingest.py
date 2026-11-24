"""Bring a downloaded farm task's output into a local shot, only under the shot's writer lock.

Farm nodes never write a local shot. The output folders of their copy (`husk.OUTPUTS`) are copied
into staging on the shot's filesystem, checked, then renamed into place, all while the lock is held.
Published sidecar frames stay write-once: an identical frame is skipped and a different one refuses
the whole ingest before anything moves. Other files replace their local counterparts. Symlinks,
special files and entries outside the output folders are refused.
"""

import os
import secrets
import shutil
from pathlib import Path

from strata_datagen import shotlock
from strata_datagen.render import husk, pack

SIDECARS = ("samples", "merged")


def _files(source: Path) -> list[Path]:
    for entry in source.iterdir():
        if entry.name not in husk.OUTPUTS or entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"{entry}: not a render output folder ({', '.join(husk.OUTPUTS)})")
    found: list[Path] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"{path}: not a regular file or folder")
        if path.is_file():
            found.append(path.relative_to(source))
    return found


def _unit(relative: Path) -> Path:
    """What moves as one: a whole published sidecar frame, or a single file."""
    return Path(*relative.parts[:2]) if relative.parts[0] in SIDECARS else relative


def _contents(path: Path) -> dict[str, bytes]:
    if path.is_file():
        return {"": path.read_bytes()}
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def _moves(staging: Path, shot: Path, unit: Path) -> bool:
    """False when the unit is already in the shot as staged; raise when it would break write-once."""
    local = shot / unit
    if not local.exists(follow_symlinks=False):
        return True
    if unit.parts[0] in SIDECARS:
        if _contents(staging / unit) != _contents(local):
            raise ValueError(f"{local}: a different published frame is already there; {pack.REMOVE}")
        return False
    if local.is_symlink() or not local.is_file():
        raise ValueError(f"{local}: not a regular file, refusing to replace it")
    return True


def ingest(shot: Path, source: Path) -> list[str]:
    """Move `source`'s output folders into `shot`; returns the files and sidecar frames moved in."""
    found = _files(source)
    units = sorted({_unit(relative) for relative in found})
    with shotlock.held(shot):
        staging = shot / pack.STAGING / f"ingest.{os.getpid()}.{secrets.token_hex(4)}"
        try:
            for relative in found:
                (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, staging / relative)
            moving = [unit for unit in units if _moves(staging, shot, unit)]
            for unit in moving:
                (shot / unit).parent.mkdir(parents=True, exist_ok=True)
                (staging / unit).replace(shot / unit)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return [unit.as_posix() for unit in moving]
