"""Drive husk over shot USD files: farm-ready job lists and a lock-holding runner that records timings."""

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from strata_datagen import files
from strata_datagen.exr import header

RENDERER = "BRAY_HdKarmaXPU"
GPU_ONLY = {"KARMA_XPU_DISABLE_EMBREE_DEVICE": "1"}


CHUNK_FRAMES = 60
OUTPUTS = ("beauty", "deep", "layers", "samples", "merged", "qa")
TASK = ("strata-datagen", "render-task")


@dataclass(frozen=True)
class Job:
    """One farm task: run `command` from `workdir`, a path relative to the build root.

    The command is `strata-datagen render-task`, never husk itself: it holds the shot lock, renders
    frames `first`..`first + count - 1` with a husk that inherits the lock, then packs, runs raw QA
    and prunes under the same hold. A shot is split into tasks of CHUNK_FRAMES frames so the per-task
    scene load and kernel compile are amortised over many frames.
    """

    shot_id: str
    workdir: str
    first: int
    count: int
    command: list[str]


def jobs(shot_dir: Path, root: Path, frames: tuple[int, ...], chunk: int) -> list[Job]:
    workdir = shot_dir.relative_to(root).as_posix()
    tasks: list[Job] = []
    for start in range(0, len(frames), chunk):
        first, count = frames[start], len(frames[start : start + chunk])
        command = [*TASK, "--shot", ".", "--first", str(first), "--count", str(count)]
        tasks.append(Job(shot_dir.name, workdir, first, count, command))
    return tasks


def write_job_list(root: Path, jobs: list[Job]) -> None:
    """jobs.json for a farm submitter and jobs.sh for a plain shell, both with relative paths."""
    files.write_text(root / "jobs.json", json.dumps([asdict(j) for j in jobs], indent=1) + "\n")
    lines = ["#!/usr/bin/env sh", "set -e", 'cd "$(dirname "$0")"']
    lines.extend(f"(cd {shlex.quote(j.workdir)} && {shlex.join(j.command)})" for j in jobs)
    files.write_text(root / "jobs.sh", "\n".join(lines) + "\n")


class _VramMonitor(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[int] = []
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
            if out:
                self.samples.append(int(out[0]))
            self.done.wait(1.0)


def _husk(shot: Path, frames: range, husk: str, env: dict[str, str], lock: int) -> int:
    """Run husk in the shot; it inherits the lock descriptor, so the shot stays locked while it runs."""
    for folder in OUTPUTS:
        (shot / folder).mkdir(parents=True, exist_ok=True)
    arguments = [
        "-R",
        RENDERER,
        "-f",
        str(frames.start),
        "-n",
        str(len(frames)),
        "-V",
        "2",
        "scene/shot.usda",
    ]
    with (shot / "render.log").open("w") as handle:
        return subprocess.run(
            [husk, *arguments],
            cwd=shot,
            env={**os.environ, **env},
            stdout=handle,
            stderr=subprocess.STDOUT,
            pass_fds=(lock,),
            check=False,
        ).returncode


def _frame_seconds(shot: Path) -> list[float]:
    times: list[float] = []
    for frame in sorted((shot / "beauty").glob("*.exr")):
        value = header(frame).get("renderTime")
        times.append(float(np.asarray(value).ravel()[0]) if value is not None else float("nan"))
    return times


def run(shot: Path, frames: range, husk: str, lock: int) -> dict[str, object]:
    """Render `frames` GPU-only and return timing records; a GPU failure is the task's failure.

    XPU's Embree CPU device drops camera samples on hair curves and puts ZBack in front of Z, so it is
    never used. `lock` is the descriptor of the caller's held shot lock, passed on to husk.
    """
    baseline = _VramMonitor()
    baseline.start()
    time.sleep(2.0)
    baseline.done.set()
    idle = min(baseline.samples) if baseline.samples else 0
    monitor = _VramMonitor()
    monitor.start()
    started = time.monotonic()
    code = _husk(shot, frames, husk, GPU_ONLY, lock)
    monitor.done.set()
    seconds = _frame_seconds(shot)
    return {
        "exit_code": code,
        "device": "gpu",
        "wall_seconds": round(time.monotonic() - started, 2),
        "frame_seconds": seconds,
        "mean_frame_seconds": float(np.nanmean(seconds)) if seconds else None,
        "vram_peak_mib": max(monitor.samples) if monitor.samples else None,
        "vram_used_by_job_mib": (max(monitor.samples) - idle) if monitor.samples else None,
    }
