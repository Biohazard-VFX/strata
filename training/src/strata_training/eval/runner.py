import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from strata_training.baselines.deep import collapse_two, fake_deep, legacy_baselines
from strata_training.data.cache import load_layers
from strata_training.data.dataset import Frame
from strata_training.data.exr import write_deep
from strata_training.data.targets import composite
from strata_training.device import configure_gpu
from strata_training.eval.boundaries import boundary_f1, directed_boundary_counts, silhouette_recall
from strata_training.eval.calibration import calibration
from strata_training.eval.metrics import mixed_errors
from strata_training.eval.predict import load_model, predict
from strata_training.eval.protocol import entries_for
from strata_training.eval.recompose import insert_plane
from strata_training.types import Array, Layers


def visible_inverse(layers: Layers) -> Array:
    trans = np.concatenate(
        (np.ones_like(layers.alpha[:1]), np.cumprod(1 - layers.alpha[:-1], axis=0))
    )
    weight = trans * layers.alpha
    return (weight * layers.inverse_depth).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-8)


def scale_align(inverse: Array, truth: Array, mask: Array) -> tuple[Array, float]:
    valid = mask & (inverse > 0) & (truth > 0)
    scale = float(np.exp(np.median(np.log(truth[valid] / inverse[valid]))))
    return inverse * scale, scale


def affine_align(inverse: Array, truth: Array) -> Array:
    p, t = inverse[::8, ::8].ravel(), truth[::8, ::8].ravel()
    valid = (p > 0) & (t > 0)
    p, t = p[valid], t[valid]
    design = np.column_stack((p, np.ones_like(p)))
    coefficients = np.linalg.lstsq(design, t, rcond=None)[0]
    error = np.abs(design @ coefficients - t)
    keep = error <= np.quantile(error, 0.8)
    coefficients = np.linalg.lstsq(design[keep], t[keep], rcond=None)[0]
    return np.maximum(inverse * max(float(coefficients[0]), 1e-8) + float(coefficients[1]), 1e-8)


@dataclass
class Collected:
    arrays: dict[str, list[Array]] = field(default_factory=dict[str, list[Array]])
    boundary: list[list[tuple[int, int, int]]] = field(
        default_factory=list[list[tuple[int, int, int]]]
    )
    aligned_boundary: list[list[tuple[int, int, int]]] = field(
        default_factory=list[list[tuple[int, int, int]]]
    )
    silhouettes: list[float] = field(default_factory=list[float])
    sums: dict[str, float] = field(default_factory=dict[str, float])
    pixels: int = 0
    frames: int = 0

    def append(self, key: str, values: Array) -> None:
        self.arrays.setdefault(key, []).append(values)

    def summary(self) -> dict[str, Any]:
        values = {k: np.concatenate(v) for k, v in self.arrays.items()}
        result: dict[str, Any] = {"mixed_pixels": int(values["alpha"].size), "frames": self.frames}
        for key in (
            "alpha",
            "front",
            "back",
            "native_alpha",
            "native_front",
            "native_back",
            "ordered",
        ):
            result[f"{key}_{'mae' if 'alpha' in key else 'median_relative_error'}"] = float(
                np.mean(values[key]) if "alpha" in key else np.median(values[key])
            )
        result.update({key: value / max(self.pixels, 1) for key, value in self.sums.items()})
        result["coverage_calibration"] = calibration(
            values["predicted_alpha"], values["target_alpha"]
        )

        def pooled(items: list[list[tuple[int, int, int]]]) -> float | None:
            counts = np.array(items).sum(axis=0)
            return boundary_f1([(int(row[0]), int(row[1]), int(row[2])) for row in counts])

        result["directed_boundary_f1_native"] = pooled(self.boundary)
        result["directed_boundary_f1_affine_aligned"] = pooled(self.aligned_boundary)
        result["silhouette_recall_z3m_proxy"] = (
            float(np.mean(self.silhouettes)) if self.silhouettes else None
        )
        return result


def collect(
    prediction: Layers,
    target: Layers,
    data: dict[str, Array],
    method: str,
    entry: dict[str, Any],
    collected: dict[tuple[str, str], Collected],
) -> None:
    historical = collapse_two(prediction)
    gt_inverse = visible_inverse(target)
    pred_inverse = visible_inverse(prediction)
    solid = ndimage.binary_erosion(
        (target.alpha[0] > 0.999) & (target.alpha[1:].sum(axis=0) < 1e-7), iterations=4
    )
    aligned, _ = scale_align(pred_inverse, gt_inverse, solid)
    affine = affine_align(pred_inverse, gt_inverse)
    trans = np.concatenate(
        (np.ones_like(target.alpha[:1]), np.cumprod(1 - target.alpha[:-1], axis=0))
    )
    foreground = (trans * target.alpha * (target.inverse_depth > 1 / 3)).sum(axis=0) > 0.1
    reconstruction = np.abs(
        composite(prediction.alpha, prediction.rgb) - data["beauty"].astype(np.float32)
    ).mean(axis=0)
    merge = np.abs(insert_plane(prediction) - insert_plane(target)).mean(axis=0)
    relative = np.abs(gt_inverse / np.maximum(pred_inverse, 1e-8) - 1)
    relative_aligned = np.abs(gt_inverse / np.maximum(aligned, 1e-8) - 1)
    delta1 = (
        np.maximum(
            gt_inverse / np.maximum(pred_inverse, 1e-8), pred_inverse / np.maximum(gt_inverse, 1e-8)
        )
        < 1.25
    )
    reference_inv = np.stack(
        (
            data["reference_inv_fg"] if "reference_inv_fg" in data else target.inverse_depth[0],
            data["reference_inv_bg"] if "reference_inv_bg" in data else target.inverse_depth[1],
        )
    )
    reference_alpha = np.stack(
        (
            data["reference_a_fg"] if "reference_a_fg" in data else target.alpha[0],
            np.ones_like(target.alpha[0]),
        )
    )
    masks = {
        "all": np.ones_like(data["mixed"], dtype=bool),
        f"{entry['category']} / fur": data["fur"],
        f"{entry['category']} / hard": ~data["fur"],
    }
    for group, region in masks.items():
        mixed = data["mixed"] & region
        if not mixed.any():
            continue
        result = collected.setdefault((group, method), Collected())
        historical_errors = mixed_errors(
            historical.inverse_depth, historical.alpha[0], reference_inv, reference_alpha, mixed
        )
        native_errors = mixed_errors(
            prediction.inverse_depth, prediction.alpha[0], target.inverse_depth, target.alpha, mixed
        )
        for key, array in zip(("alpha", "front", "back"), historical_errors, strict=True):
            result.append(key, array)
        for key, array in zip(
            ("native_alpha", "native_front", "native_back"), native_errors, strict=True
        ):
            result.append(key, array)
        valid = target.valid & mixed[None]
        result.append(
            "ordered", np.abs(prediction.inverse_depth[valid] / target.inverse_depth[valid] - 1)
        )
        result.append("predicted_alpha", historical.alpha[0][mixed])
        result.append("target_alpha", reference_alpha[0][mixed])
        for key, array in {
            "composite_mae": reconstruction,
            "insert_z3m_mae": merge,
            "depth_absrel": relative,
            "depth_absrel_scale_aligned": relative_aligned,
            "depth_delta1": delta1,
        }.items():
            result.sums[key] = result.sums.get(key, 0.0) + float(array[region].sum())
        result.pixels += int(region.sum())
        result.frames += 1
        result.boundary.append(directed_boundary_counts(pred_inverse, gt_inverse, region))
        result.aligned_boundary.append(directed_boundary_counts(affine, gt_inverse, region))
        recall = silhouette_recall(pred_inverse, foreground, region)
        if recall is not None:
            result.silhouettes.append(recall)


def evaluate(
    index: Path,
    protocol: Path,
    output: Path,
    checkpoints: list[Path],
    legacy_results: Path | None = None,
    export_frames: int = 1,
) -> dict[str, Any]:
    entries = entries_for(index, protocol, "test")
    output.mkdir(parents=True, exist_ok=True)
    collected: dict[tuple[str, str], Collected] = {}
    for entry in entries:
        target, data = load_layers(Path(entry["cache"]))
        if legacy_results and entry["legacy_data"]:
            predictions = legacy_baselines(entry, data["beauty"].astype(np.float32), legacy_results)
            for method, prediction in predictions.items():
                collect(prediction, target, data, method, entry, collected)
        print(json.dumps({"baseline_frame": entry["id"]}), flush=True)
    if checkpoints:
        configure_gpu()
    for checkpoint in checkpoints:
        model, config = load_model(checkpoint)
        method = f"{config['preset']} {config['mode']} {config['scale_mode']}"
        fake_method = f"fake {config['preset']} {config['scale_mode']}"
        include_baseline = not any(key[1] == fake_method for key in collected)
        predictions_dir = output / checkpoint.parent.name
        predictions_dir.mkdir(exist_ok=True)
        for i, entry in enumerate(entries):
            frame = Frame(entry, Path(config["features"]), config["scale_mode"])
            target, data = load_layers(Path(entry["cache"]))
            prediction = predict(model, config, frame)
            collect(prediction, target, data, method, entry, collected)
            baseline = fake_deep(frame.base, data["beauty"].astype(np.float32))
            if include_baseline:
                collect(baseline, target, data, fake_method, entry, collected)
            np.savez_compressed(
                predictions_dir / f"{entry['id']}.npz",
                inverse_depth=prediction.inverse_depth,
                alpha=prediction.alpha,
                rgb=prediction.rgb,
            )
            if i < export_frames:
                write_deep(predictions_dir / f"{entry['id']}.exr", prediction)
            print(json.dumps({"method": method, "frame": entry["id"]}), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = [
        {"group": group, "method": method, **values.summary()}
        for (group, method), values in sorted(collected.items())
    ]
    report = {
        "summary": summary,
        "test_ids": [e["id"] for e in entries],
        "protocol": json.loads(protocol.read_text()),
        "temporal_jitter": None,
        "temporal_reason": "v004 contains isolated single-frame shots, no GT tracks",
        "boundary_protocol": "Directed adjacent-pixel ratios at 5,10,15,20,25%, threshold-proportional weighting. Exact native-pixel matching, four directed orientations. Affine-aligned results separate. Silhouette proxy: coverage at Z<3m thresholded at .1; scanline connected-run NMS. Not a published-benchmark reproduction.",
        "baseline_color_policy": "Analytic/probe lack RGB predictions; reuse observed beauty at every layer. Composite zero is tautological. Insert-plane error tests geometry but is not hidden-color recovery.",
        "historical_protocol": "Collapse K predictions to two at largest visible log gap (.003 visible-contribution cutoff), compare with original v004 two-layer GT/masks. Native K metrics are separate; probe remains oracle-conditioned.",
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if checkpoints else 0,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report
