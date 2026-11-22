"""Drive husk over shot USD files: farm-ready job lists and a local runner that records timings."""

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
OUTPUTS = ("beauty", "deep", "layers", "samples", "qa")


@dataclass(frozen=True)
class Job:
    """One farm task: run `command` (husk), then each `post` command in order, from `workdir`.

    Post commands pack targets, run raw QA on the task's frames and only then prune off-stride raw
    deep; a failing command stops the chain, so unaccepted frames keep their raw deep. Paths are
    relative to the build root. A shot is split into tasks of CHUNK_FRAMES frames so the per-task
    scene load and kernel compile are amortised over many frames.
    """

    shot_id: str
    workdir: str
    command: list[str]
    outputs: list[str]
    post: list[list[str]]


POST_STEPS = ("pack", "accept", "prune")


def jobs(shot_dir: Path, root: Path, frames: tuple[int, ...], chunk: int) -> list[Job]:
    workdir = shot_dir.relative_to(root).as_posix()
    tasks: list[Job] = []
    for start in range(0, len(frames), chunk):
        first, count = frames[start], len(frames[start : start + chunk])
        command = ["husk", "-R", RENDERER, "-f", str(first), "-n", str(count), "-V", "2", "scene/shot.usda"]
        post = [
            ["strata-datagen", step, "--shot", ".", "--first", str(first), "--count", str(count)]
            for step in POST_STEPS
        ]
        tasks.append(Job(shot_dir.name, workdir, command, list(OUTPUTS), post))
    return tasks


def write_job_list(root: Path, jobs: list[Job]) -> None:
    """jobs.json for a farm submitter and jobs.sh for a plain shell, both with relative paths."""
    files.write_text(root / "jobs.json", json.dumps([asdict(j) for j in jobs], indent=1) + "\n")
    lines = ["#!/bin/sh", "set -e", 'cd "$(dirname "$0")"']
    lines.extend(
        f"(cd {shlex.quote(j.workdir)} && mkdir -p {' '.join(j.outputs)} && "
        + " && ".join(shlex.join(c) for c in (j.command, *j.post))
        + ")"
        for j in jobs
    )
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


def _husk(root: Path, j: Job, husk: str, env: dict[str, str], log: Path) -> int:
    cwd = root / j.workdir
    for folder in j.outputs:
        (cwd / folder).mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        return subprocess.run(
            [husk, *j.command[1:]],
            cwd=cwd,
            env={**os.environ, **env},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def _frame_seconds(shot: Path) -> list[float]:
    times: list[float] = []
    for frame in sorted((shot / "beauty").glob("*.exr")):
        value = header(frame).get("renderTime")
        times.append(float(np.asarray(value).ravel()[0]) if value is not None else float("nan"))
    return times


def run(root: Path, j: Job, husk: str) -> dict[str, object]:
    """Render one shot GPU-only, falling back to XPU with the CPU device; returns timing records."""
    baseline = _VramMonitor()
    baseline.start()
    time.sleep(2.0)
    baseline.done.set()
    idle = min(baseline.samples) if baseline.samples else 0
    monitor = _VramMonitor()
    monitor.start()
    started = time.monotonic()
    log = root / j.workdir / "render.log"
    device = "gpu"
    code = _husk(root, j, husk, GPU_ONLY, log)
    if code != 0:
        device = "gpu+cpu"
        code = _husk(root, j, husk, {}, log)
    monitor.done.set()
    frames = _frame_seconds(root / j.workdir)
    return {
        "exit_code": code,
        "device": device,
        "wall_seconds": round(time.monotonic() - started, 2),
        "frame_seconds": frames,
        "mean_frame_seconds": float(np.nanmean(frames)) if frames else None,
        "vram_peak_mib": max(monitor.samples) if monitor.samples else None,
        "vram_used_by_job_mib": (max(monitor.samples) - idle) if monitor.samples else None,
    }
