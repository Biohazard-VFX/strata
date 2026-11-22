"""Per-shot acceptance checks; a shot that fails any check is rejected from the dataset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from strata_datagen.exr import header, read_deep, read_flat
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
    "valid_id_fraction": (0.999, 1.0),
    "camera_error": (0.0, 1e-3),
}
COVERED = 0.999
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
    ids = instance_ids(deep)
    valid = np.zeros(ids.size, bool)
    for low, high in ranges:
        valid |= (ids >= low) & (ids < high)
    covered = flat.alpha >= COVERED
    residual = np.stack([beauty[c] - flat.rgba[..., i] for i, c in enumerate("RGB")], -1)
    return {
        "mean_samples": float(deep.counts.mean()),
        "max_samples": float(deep.counts.max()),
        "bad_depth": float((~np.isfinite(z) | (z <= 0) | (zback < z - 1e-4)).sum()),
        "partial_share": float(flat.partial.mean()),
        "flatten_error": float(np.abs(residual[covered]).mean()) if covered.any() else 0.0,
        "flatten_excess": float(np.maximum(-residual, 0.0).mean()),
        "valid_id_fraction": float(valid.mean()) if valid.size else 1.0,
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
    if not beauty_path.exists() or not deep_path.exists():
        return FrameResult(frame, {}, ["missing beauty or deep"])
    beauty = read_flat(beauty_path)
    metrics = _beauty_metrics(beauty)
    metrics.update(deep_metrics(deep_path, beauty, ranges))
    metrics["camera_error"] = _camera_error(beauty_path, record)
    failures = [f"{k}={metrics[k]:.4g}" for k in ("nonfinite", "bad_depth") if metrics[k] > 0]
    failures += [
        f"{k}={metrics[k]:.4g} outside {low}..{high}"
        for k, (low, high) in LIMITS.items()
        if not low <= metrics[k] <= high
    ]
    return FrameResult(frame, metrics, failures)


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
