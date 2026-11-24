import json
from pathlib import Path
from typing import Any

import numpy as np
from strata_dataset.records import VERSION, errors
from strata_dataset.sampling import noise_varies_per_frame

from strata_training.types import Array

CONTRACT = VERSION


def camera_frames(shot: Path) -> dict[int, dict[str, Any]]:
    camera: dict[str, Any] = json.loads((shot / "camera.json").read_text())
    manifest: dict[str, Any] = json.loads((shot / "manifest.json").read_text())
    violations = errors(manifest, camera)
    if violations:
        raise ValueError(f"Unsupported dataset records in {shot}: {violations}")
    records: list[dict[str, Any]] = camera["frames"]
    frames: dict[int, dict[str, Any]] = {}
    for record in records:
        frame = record["frame"]
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 1 or frame in frames:
            raise ValueError("Camera frame identifiers must be distinct positive integers")
        frames[frame] = {
            **record,
            "noise_varies_per_frame": noise_varies_per_frame(manifest["render"]["sampling_seed"]),
        }
    return frames


def intrinsics(record: dict[str, Any]) -> Array:
    return np.array(
        [[record["fx"], 0, record["cx"]], [0, record["fy"], record["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )


def category(record: dict[str, Any]) -> str:
    blur = record["shutter_close"] > record["shutter_open"]
    dof = record["fstop"] > 0
    return "mb+dof" if blur and dof else "mb" if blur else "dof" if dof else "sharp"
