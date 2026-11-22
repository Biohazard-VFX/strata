"""Per-shot acceptance checks; a shot that fails any check is rejected from the dataset.

Raw deep is checked before anything is removed: `accept` records each frame's raw QA result with the
identity of every input it validated (`inputs`), `prune` removes off-stride raw deep only from
accepted frames whose inputs are unchanged, and `check_shot` re-checks every packed target and holds
pruned frames to a recorded acceptance whose inputs, all but the removed deep, still match.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import OpenEXR
from numpy.typing import NDArray

from strata_contract import samples as contract_samples
from strata_datagen import files
from strata_datagen.exr import Deep, header, read_deep, read_flat
from strata_datagen.generators.lighting import BACKDROP_M, SKY_ID
from strata_datagen.qa.deep import flatten, instance_ids
from strata_datagen.render import pack

VALIDATION = "raw-qa-1"
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
    "valid_id_fraction": (0.99995, 1.0),
    "coverage_channel": (1.0, 1.0),
    "coverage_alpha_error": (0.0, 2.0**-11),
    "sky_id_in_front": (0.0, 1e-5),
    "camera_error": (0.0, 1e-3),
    "intrinsics_error": (0.0, 1e-4),
    "target_format": (1.0, 1.0),
    "target_error": (0.0, 0.005),
    "target_excess": (0.0, 0.005),
    "target_mismatch": (0.0, 0.0),
    "samples_mismatch": (0.0, 0.0),
}
COVERED = 0.999
SKY_MIN_Z_M = 0.25 * BACKDROP_M
STEP_LIMITS = {"position_m": 0.25, "rotation_deg": 2.0, "focal_ratio": 0.02, "fstop": 0.5, "focus_m": 0.5}
FLIP_Z = np.diag([1.0, 1.0, -1.0, 1.0])
DEEP_KEYS = (
    "mean_samples",
    "max_samples",
    "bad_depth",
    "partial_share",
    "flatten_error",
    "uncovered",
    "flatten_excess",
    "valid_id_fraction",
    "coverage_channel",
    "coverage_alpha_error",
    "sky_id_in_front",
)
LAYER_NAMES = tuple(f"L{k}.{c}" for k in range(4) for c in ("invdepth", "A", "R", "G", "B"))


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
    deep: Deep, beauty: dict[str, NDArray[np.float32]], ranges: list[tuple[int, int]]
) -> dict[str, float]:
    flat = flatten(deep)
    z, zback = deep.channels["Z"], deep.channels["ZBack"]
    valid = np.zeros(int(deep.counts.sum()), bool)
    near_sky = 0.0
    alpha_error = 1.0
    if "coverage" in deep.channels:
        ids = instance_ids(deep)
        for low, high in ranges:
            valid |= (ids >= low) & (ids < high)
        near_sky = float(((ids == SKY_ID) & (z < SKY_MIN_Z_M)).mean())
        difference = deep.channels["coverage"].astype(np.float64) - deep.channels["A"]
        alpha_error = float(np.abs(difference).max(initial=0.0))
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
        "coverage_alpha_error": alpha_error,
        "sky_id_in_front": near_sky,
    }


def _layer_dtypes(path: Path) -> dict[str, np.dtype[np.generic]]:
    part = OpenEXR.File(str(path), separate_channels=True).parts[0]
    return {name: channel.pixels.dtype for name, channel in part.channels.items()}


def target_metrics(path: Path, beauty: dict[str, NDArray[np.float32]]) -> dict[str, float]:
    """Packed targets on their own: float32, finite, in range, and compositing to the beauty."""
    dtypes = _layer_dtypes(path)
    targets = read_flat(path)
    if any(dtypes.get(name) != np.dtype(np.float32) for name in LAYER_NAMES):
        return {"target_format": 0.0}
    values = np.stack([targets[name] for name in LAYER_NAMES]).reshape(4, 5, *targets["L0.A"].shape)
    alpha, inverse = values[:, 1].astype(np.float64), values[:, 0]
    in_range = np.isfinite(values).all() and np.all((alpha >= 0) & (alpha <= 1)) and np.all(inverse >= 0)
    transmitted = np.concatenate([np.ones_like(alpha[:1]), np.cumprod(1.0 - alpha[:-1], axis=0)])
    composite = (transmitted[:, None] * values[:, 2:]).sum(axis=0)
    residual = np.stack([beauty[c] for c in "RGB"]) - composite
    return {
        "target_format": float(in_range),
        "target_error": float(np.abs(residual).mean()),
        "target_excess": float(np.maximum(-residual, 0.0).mean()),
    }


def _target_mismatch(path: Path, frame: pack.Samples) -> float:
    stored = read_flat(path)
    expected = pack.layer_channels(pack.layers(frame))
    return float(max(np.abs(stored[name] - value).max(initial=0.0) for name, value in expected.items()))


def _samples_mismatch(directory: Path, frame: pack.Samples, beauty_path: Path) -> tuple[float, list[str]]:
    errors = contract_samples.errors(directory, frame.shape)
    if errors:
        return 1.0, errors
    stored = contract_samples.load(directory)
    expected: dict[str, NDArray[np.generic]] = {**frame.arrays(), **pack.images(beauty_path)}
    return float(any(not np.array_equal(stored[k], v) for k, v in expected.items())), []


def _camera_error(path: Path, record: dict[str, object]) -> float:
    world_to_camera = np.asarray(record["world_to_camera"], dtype=np.float64)
    husk = np.asarray(header(path)["worldToCamera_d"], dtype=np.float64)
    return float(np.abs((FLIP_Z @ world_to_camera).T - husk).max())


def _intrinsics_error(path: Path, record: dict[str, object]) -> float:
    """Recorded intrinsics against the projection husk rendered with (screen window -1..1)."""
    found = header(path)
    projection = np.linalg.inv(np.asarray(found["worldToCamera_d"], np.float64)) @ np.asarray(
        found["worldToNDC_d"], np.float64
    )
    width, height = cast("int", record["width"]), cast("int", record["height"])
    rendered = {
        "fx": projection[0, 0] * width / 2.0,
        "fy": projection[1, 1] * height / 2.0,
        "cx": (projection[2, 0] + 1.0) * width / 2.0,
        "cy": (1.0 - projection[2, 1]) * height / 2.0,
    }
    scale = {"fx": rendered["fx"], "fy": rendered["fy"], "cx": width, "cy": height}
    return max(abs(cast("float", record[k]) - v) / scale[k] for k, v in rendered.items())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inputs(shot: Path, frame: int) -> dict[str, str | None]:
    """Identity of everything raw QA of `frame` reads; manifest `timings` is bookkeeping, not input."""
    paths, plan = _paths(shot, frame), pack.ShotFrames.load(shot)
    manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
    camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
    records = cast("list[dict[str, object]]", camera.pop("frames"))
    return {
        "validation": _canonical({"version": VALIDATION, "limits": LIMITS, "deep_keys": DEEP_KEYS}),
        "manifest": _canonical({k: v for k, v in manifest.items() if k != "timings"}),
        "camera": _canonical({**camera, "frame": [r for r in records if r["frame"] == frame]}),
        "beauty": digest(paths["beauty"]) if paths["beauty"].exists() else None,
        "deep": digest(paths["deep"]) if paths["deep"].exists() else None,
        "targets": digest(paths["layers"]) if paths["layers"].exists() else None,
        "samples": (
            contract_samples.recorded_identity(plan.sample_dir(shot, frame))
            if plan.recomposition(frame)
            else None
        ),
    }


def changed_inputs(record: dict[str, object], current: dict[str, str | None]) -> list[str]:
    recorded = cast("dict[str, str | None]", record.get("inputs", {}))
    return [name for name, value in current.items() if name not in recorded or recorded[name] != value]


def _paths(shot: Path, frame: int) -> dict[str, Path]:
    name = f"{frame:04d}.exr"
    return {folder: shot / folder / name for folder in ("beauty", "deep", "layers")} | {
        "acceptance": shot / "qa" / f"{frame:04d}.json"
    }


def _raw_metrics(
    shot: Path, frame: int, beauty: dict[str, NDArray[np.float32]], ranges: list[tuple[int, int]]
) -> tuple[dict[str, float], list[str]]:
    paths, plan = _paths(shot, frame), pack.ShotFrames.load(shot)
    deep = read_deep(paths["deep"])
    metrics = deep_metrics(deep, beauty, ranges)
    frame_samples = pack.samples(deep)
    failures: list[str] = []
    if paths["layers"].exists():
        metrics["target_mismatch"] = _target_mismatch(paths["layers"], frame_samples)
    if plan.recomposition(frame):
        mismatch, failures = _samples_mismatch(plan.sample_dir(shot, frame), frame_samples, paths["beauty"])
        metrics["samples_mismatch"] = mismatch
    return metrics, failures


def _accepted(shot: Path, frame: int) -> tuple[dict[str, float], list[str]]:
    """A pruned frame stands on its recorded raw acceptance, valid only for the targets it saw."""
    paths, plan = _paths(shot, frame), pack.ShotFrames.load(shot)
    if plan.recomposition(frame):
        return {}, ["recomposition frame lost its raw deep"]
    if not paths["acceptance"].exists():
        return {}, ["no raw deep and no raw acceptance"]
    record = cast("dict[str, object]", json.loads(paths["acceptance"].read_text()))
    failures = [] if record["passed"] else ["raw acceptance failed"]
    changed = [name for name in changed_inputs(record, inputs(shot, frame)) if name != "deep"]
    failures += [f"{name} changed since raw acceptance" for name in changed]
    metrics = cast("dict[str, float]", record["metrics"])
    return {k: v for k, v in metrics.items() if k in DEEP_KEYS}, failures


def check_frame(
    shot: Path, frame: int, record: dict[str, object], ranges: list[tuple[int, int]]
) -> FrameResult:
    paths = _paths(shot, frame)
    if not paths["beauty"].exists():
        return FrameResult(frame, {}, ["missing beauty"])
    beauty = read_flat(paths["beauty"])
    metrics = _beauty_metrics(beauty)
    failures: list[str] = []
    if paths["layers"].exists():
        metrics.update(target_metrics(paths["layers"], beauty))
    else:
        failures.append("missing layers")
    raw, found = (
        _raw_metrics(shot, frame, beauty, ranges) if paths["deep"].exists() else _accepted(shot, frame)
    )
    metrics.update(raw)
    failures += found
    metrics["camera_error"] = _camera_error(paths["beauty"], record)
    metrics["intrinsics_error"] = _intrinsics_error(paths["beauty"], record)
    failures += [f"{k}={metrics[k]:.4g}" for k in ("nonfinite", "bad_depth") if metrics.get(k, 0.0) > 0]
    failures += [
        f"{k}={metrics[k]:.4g} outside {low}..{high}"
        for k, (low, high) in LIMITS.items()
        if k in metrics and not low <= metrics[k] <= high
    ]
    return FrameResult(frame, metrics, failures)


def _records(shot: Path) -> tuple[dict[int, dict[str, object]], list[tuple[int, int]]]:
    manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
    camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
    frames = cast("list[dict[str, object]]", camera["frames"])
    return {cast("int", f["frame"]): f for f in frames}, _id_ranges(manifest)


def accept(shot: Path, frames: range) -> list[FrameResult]:
    """Raw QA of rendered, packed frames; records each result with the identity of its inputs.

    A frame whose raw deep is already pruned is checked against its recorded acceptance, never re-recorded.
    """
    records, ranges = _records(shot)
    results: list[FrameResult] = []
    for frame in frames:
        paths = _paths(shot, frame)
        checked = inputs(shot, frame)
        result = check_frame(shot, frame, records[frame], ranges)
        results.append(result)
        if not paths["deep"].exists():
            continue
        files.write_text(
            paths["acceptance"],
            json.dumps(
                {
                    "frame": frame,
                    "passed": not result.failures,
                    "failures": result.failures,
                    "metrics": result.metrics,
                    "inputs": checked,
                },
                indent=1,
            )
            + "\n",
        )
    return results


def prune(shot: Path, frames: range) -> list[int]:
    """Delete off-stride raw deep of accepted frames whose every input is unchanged since QA."""
    plan = pack.ShotFrames.load(shot)
    removable: list[Path] = []
    refused: list[str] = []
    for frame in frames:
        paths = _paths(shot, frame)
        if plan.recomposition(frame) or not paths["deep"].exists():
            continue
        record: dict[str, object] = (
            cast("dict[str, object]", json.loads(paths["acceptance"].read_text()))
            if paths["acceptance"].exists()
            else {"passed": False}
        )
        changed = changed_inputs(record, inputs(shot, frame))
        if not record["passed"] or changed or not paths["layers"].exists():
            refused.append(f"{frame} ({', '.join(changed) or 'not accepted'})")
        else:
            removable.append(paths["deep"])
    if refused:
        raise ValueError(f"raw deep kept: frames {'; '.join(refused)} have no matching passed acceptance")
    for path in removable:
        path.unlink()
    return [int(p.stem) for p in removable]


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


def camera_consistency(camera: dict[str, object]) -> list[str]:
    frames = cast("list[dict[str, object]]", camera["frames"])
    failures: list[str] = []
    for record in frames:
        m = np.asarray(record["world_to_camera"], dtype=np.float64)
        rotation = m[:3, :3]
        width = cast("int", record["width"])
        fx, fy = cast("float", record["fx"]), cast("float", record["fy"])
        expected = cast("float", record["focal_length_mm"]) / cast("float", record["sensor_width_mm"]) * width
        if not np.isfinite(m).all() or np.abs(rotation @ rotation.T - np.eye(3)).max() > 1e-5:
            failures.append(f"frame {record['frame']}: world_to_camera is not a rigid transform")
        if abs(fx - expected) > 1e-3 * expected:
            failures.append(f"frame {record['frame']}: fx disagrees with focal length")
        if abs(fy - fx) > 1e-6 * fx:
            failures.append(f"frame {record['frame']}: fy differs from fx with square pixels")
    steps = camera_steps(frames)
    failures += [
        f"camera {k} step {v:.4g} > {STEP_LIMITS[k]}" for k, v in steps.items() if v > STEP_LIMITS[k]
    ]
    return failures


def check_shot(shot: Path) -> dict[str, object]:
    manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
    camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
    frames = cast("list[dict[str, object]]", camera["frames"])
    failures = camera_consistency(camera)
    first, last = cast("list[int]", cast("dict[str, object]", manifest["render"])["frames"])
    if [f["frame"] for f in frames] != list(range(first, last + 1)):
        failures.append(f"camera.json frames do not cover the rendered range {first}..{last}")
    ranges = _id_ranges(manifest)
    results = [check_frame(shot, cast("int", f["frame"]), f, ranges) for f in frames]
    failures += [f"frame {r.frame}: {f}" for r in results for f in r.failures]
    if not any(_paths(shot, r.frame)["deep"].exists() for r in results):
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
