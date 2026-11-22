"""Per-shot acceptance checks; a shot that fails any check is rejected from the dataset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from strata_datagen.exr import header, read_deep, read_flat
from strata_datagen.generators.lighting import BACKDROP_M, SKY_ID
from strata_datagen.qa.deep import flatten, instance_ids

LUMA = (0.2126, 0.7152, 0.0722)
LIMITS = {
    "median_luma": (0.02, 1.5),
    "clipped_fraction": (0.0, 0.02),
    "mean_samples": (1.0, 48.0),
    "max_samples": (1.0, 1024.0),
    "partial_share": (0.005, 1.0),
    "flatten_error": (0.0, 0.005),
    "flatten_excess": (0.0, 0.005),
    "uncovered": (0.0, 0.0),
    "valid_id_fraction": (0.999, 1.0),
    "coverage_channel": (1.0, 1.0),
    "sky_id_in_front": (0.0, 1e-5),
    "camera_error": (0.0, 1e-3),
}
COVERED = 0.999
SKY_MIN_Z_M = 0.25 * BACKDROP_M
STEP_LIMITS = {"position_m": 0.25, "rotation_deg": 2.0, "focal_ratio": 0.02, "fstop": 0.5, "focus_m": 0.5}
FLIP_Z = np.diag([1.0, 1.0, -1.0, 1.0])


@dataclass(frozen=True)
class FrameResult:
    frame: int
    metrics: dict[str, float]
    failures: list[str]


def _beauty_metrics(beauty: dict[str, NDArray[np.float32]]) -> dict[str, float]:
    stacked = np.stack(list(beauty.values()))
    luma = beauty["R"] * LUMA[0] + beauty["G"] * LUMA[1] + beauty["B"] * LUMA[2]
    return {
        "nonfinite": float((~np.isfinite(stacked)).sum()),
        "median_luma": float(np.median(luma)),
        "clipped_fraction": float((luma > 16.0).mean()),
    }


def _id_ranges(manifest: dict[str, object]) -> list[tuple[int, int]]:
    ranges = cast("list[dict[str, int]]", manifest["instance_ids"])
    return [(r["first_id"], r["first_id"] + r["count"]) for r in ranges]


def deep_metrics(
    path: Path, beauty: dict[str, NDArray[np.float32]], ranges: list[tuple[int, int]]
) -> dict[str, float]:
    deep = read_deep(path)
    flat = flatten(deep)
    z, zback = deep.channels["Z"], deep.channels["ZBack"]
    valid = np.zeros(int(deep.counts.sum()), bool)
    near_sky = 0.0
    if "coverage" in deep.channels:
        ids = instance_ids(deep)
        for low, high in ranges:
            valid |= (ids >= low) & (ids < high)
        near_sky = float(((ids == SKY_ID) & (z < SKY_MIN_Z_M)).mean())
    residual = np.stack([beauty[c] - flat.rgba[..., i] for i, c in enumerate("RGB")], -1)
    return {
        "mean_samples": float(deep.counts.mean()),
        "max_samples": float(deep.counts.max()),
        "bad_depth": float((~np.isfinite(z) | (z <= 0) | (zback < z * (1.0 - 1e-6) - 1e-4)).sum()),
        "partial_share": float(flat.partial.mean()),
        "flatten_error": float(np.abs(residual).mean()),
        "uncovered": float((flat.alpha < COVERED).mean()),
        "flatten_excess": float(np.maximum(-residual, 0.0).mean()),
        "valid_id_fraction": float(valid.mean()) if valid.size else 1.0,
        "coverage_channel": float("coverage" in deep.channels),
        "sky_id_in_front": near_sky,
    }


def _camera_error(path: Path, record: dict[str, object]) -> float:
    world_to_camera = np.asarray(record["world_to_camera"], dtype=np.float64)
    husk = np.asarray(header(path)["worldToCamera_d"], dtype=np.float64)
    return float(np.abs((FLIP_Z @ world_to_camera).T - husk).max())


def check_frame(
    shot: Path, frame: int, record: dict[str, object], ranges: list[tuple[int, int]]
) -> FrameResult:
    name = f"{frame:04d}.exr"
    beauty_path, deep_path = shot / "beauty" / name, shot / "deep" / name
    if not beauty_path.exists():
        return FrameResult(frame, {}, ["missing beauty"])
    beauty = read_flat(beauty_path)
    metrics = _beauty_metrics(beauty)
    if deep_path.exists():
        metrics.update(deep_metrics(deep_path, beauty, ranges))
    metrics["camera_error"] = _camera_error(beauty_path, record)
    failures = [f"{k}={metrics[k]:.4g}" for k in ("nonfinite", "bad_depth") if metrics.get(k, 0.0) > 0]
    failures += [
        f"{k}={metrics[k]:.4g} outside {low}..{high}"
        for k, (low, high) in LIMITS.items()
        if k in metrics and not low <= metrics[k] <= high
    ]
    return FrameResult(frame, metrics, failures)


def camera_steps(frames: list[dict[str, object]]) -> dict[str, float]:
    """Largest change between consecutive frames of each camera quantity; shots must have no jumps."""
    c2w = np.linalg.inv(np.array([np.asarray(f["world_to_camera"], np.float64) for f in frames]))
    rotation = np.einsum("nij,nkj->nik", c2w[1:, :3, :3], c2w[:-1, :3, :3])
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    lens = np.array(
        [[cast("float", f[k]) for k in ("focal_length_mm", "fstop", "focus_distance_m")] for f in frames]
    )
    return {
        "position_m": float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).max(initial=0.0)),
        "rotation_deg": float(angle.max(initial=0.0)),
        "focal_ratio": float(np.abs(np.diff(np.log(lens[:, 0]))).max(initial=0.0)),
        "fstop": float(np.abs(np.diff(lens[:, 1])).max(initial=0.0)),
        "focus_m": float(np.abs(np.diff(lens[:, 2])).max(initial=0.0)),
    }


def _camera_consistency(camera: dict[str, object]) -> list[str]:
    frames = cast("list[dict[str, object]]", camera["frames"])
    failures: list[str] = []
    for record in frames:
        m = np.asarray(record["world_to_camera"], dtype=np.float64)
        rotation = m[:3, :3]
        width = cast("int", record["width"])
        fx = cast("float", record["fx"])
        expected = cast("float", record["focal_length_mm"]) / cast("float", record["sensor_width_mm"]) * width
        if not np.isfinite(m).all() or np.abs(rotation @ rotation.T - np.eye(3)).max() > 1e-5:
            failures.append(f"frame {record['frame']}: world_to_camera is not a rigid transform")
        if abs(fx - expected) > 1e-3 * expected:
            failures.append(f"frame {record['frame']}: fx disagrees with focal length")
    steps = camera_steps(frames)
    failures += [
        f"camera {k} step {v:.4g} > {STEP_LIMITS[k]}" for k, v in steps.items() if v > STEP_LIMITS[k]
    ]
    return failures


def check_shot(shot: Path) -> dict[str, object]:
    manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
    camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
    frames = cast("list[dict[str, object]]", camera["frames"])
    failures = _camera_consistency(camera)
    first, last = cast("list[int]", cast("dict[str, object]", manifest["render"])["frames"])
    if [f["frame"] for f in frames] != list(range(first, last + 1)):
        failures.append(f"camera.json frames do not cover the rendered range {first}..{last}")
    ranges = _id_ranges(manifest)
    results = [check_frame(shot, cast("int", f["frame"]), f, ranges) for f in frames]
    failures += [f"frame {r.frame}: {f}" for r in results for f in r.failures]
    if not any("mean_samples" in r.metrics for r in results):
        failures.append("no raw deep frame")
    summary = {
        k: float(np.mean([r.metrics[k] for r in results if k in r.metrics]))
        for k in sorted({k for r in results for k in r.metrics})
    }
    return {
        "shot_id": shot.name,
        "passed": not failures,
        "failures": failures,
        "mean_metrics": summary,
        "frames": [{"frame": r.frame, **r.metrics} for r in results],
    }
