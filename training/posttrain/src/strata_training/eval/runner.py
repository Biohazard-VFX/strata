import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from strata_training.baselines.cached import cached_baselines
from strata_training.baselines.deep import fake_deep
from strata_training.data.cache import load_layers
from strata_training.data.dataset import Frame
from strata_training.data.exr import write_deep
from strata_training.data.targets import composite, visible_inverse
from strata_training.eval.boundaries import boundary_f1, directed_boundary_counts, silhouette_recall
from strata_training.eval.calibration import calibration
from strata_training.eval.metrics import mixed_errors
from strata_training.eval.predict import load_model, predict
from strata_training.eval.protocol import entries_for
from strata_training.eval.recompose import insert_plane
from strata_training.eval.temporal import sequence_jitter
from strata_training.io import save_arrays, write_json
from strata_training.types import Array, Layers


def scale_align(inverse: Array, truth: Array, mask: Array) -> tuple[Array, float]:
    valid = mask & (truth > 0)
    if not valid.any():
        valid = truth > 0
    scale = float(np.exp(np.median(np.log(truth[valid] / np.maximum(inverse[valid], 1e-8)))))
    return inverse * scale, scale


def affine_align(inverse: Array, truth: Array) -> Array:
    p, t = inverse[::8, ::8].ravel(), truth[::8, ::8].ravel()
    valid = (p > 0) & (t > 0)
    p, t = p[valid], t[valid]
    if not p.size:
        return np.maximum(inverse, 1e-8)
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
    geometry_pixels: int = 0
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
            result[f"{key}_{'mae' if 'alpha' in key else 'median_relative_error'}"] = (
                float(
                    np.mean(values[key], dtype=np.float64)
                    if "alpha" in key
                    else np.median(values[key])
                )
                if values[key].size
                else None
            )
        for key, value in self.sums.items():
            count = self.geometry_pixels if key.startswith("depth_") else self.pixels
            result[key] = value / count if count else None
        result["coverage_calibration"] = calibration(
            values["predicted_alpha"], values["target_alpha"]
        )

        def pooled(items: list[list[tuple[int, int, int]]]) -> float | None:
            counts = np.array(items).sum(axis=0)
            return boundary_f1([(int(row[0]), int(row[1]), int(row[2])) for row in counts])

        for key in ("edge_composite", "edge_insert"):
            result[f"{key}_mae"] = float(values[key].mean()) if values[key].size else None
        result["native_coverage_calibration"] = calibration(
            values["native_predicted_alpha"], values["native_target_alpha"]
        )
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
    foreground = data.get("foreground", foreground)
    reconstruction = np.abs(
        composite(prediction.alpha, prediction.rgb) - data["beauty"].astype(np.float32)
    ).mean(axis=0)
    merge = np.abs(insert_plane(prediction) - insert_plane(target)).mean(axis=0)
    edge = ndimage.binary_dilation(data["mixed"], iterations=2)
    layer_rgb = np.abs(prediction.rgb - target.rgb).mean(axis=(0, 1))
    count_error = np.abs((prediction.alpha > 0.03).sum(axis=0) - (target.alpha > 0.03).sum(axis=0))
    reversed_order = (prediction.inverse_depth[:-1] < prediction.inverse_depth[1:]).any(axis=0)
    relative = np.abs(gt_inverse / np.maximum(pred_inverse, 1e-8) - 1)
    relative_aligned = np.abs(gt_inverse / np.maximum(aligned, 1e-8) - 1)
    delta1 = (
        np.maximum(
            gt_inverse / np.maximum(pred_inverse, 1e-8), pred_inverse / np.maximum(gt_inverse, 1e-8)
        )
        < 1.25
    )
    positions = np.arange(target.alpha.shape[0], dtype=np.int64)[:, None, None]
    last = np.where(target.valid, positions, 0).max(axis=0)[None]
    target_back: Array = np.take_along_axis(target.inverse_depth, last, axis=0)[0]
    predicted_back = np.take_along_axis(prediction.inverse_depth, last, axis=0)[0]
    historical_inverse = np.stack((prediction.inverse_depth[0], predicted_back))
    reference_inv = np.stack(
        (
            data["reference_inv_fg"] if "reference_inv_fg" in data else target.inverse_depth[0],
            data.get("reference_inv_bg", target_back),
        )
    )
    reference_alpha = np.stack(
        (
            data["reference_a_fg"] if "reference_a_fg" in data else target.alpha[0],
            (last[0] > 0).astype(np.float32),
        )
    )
    masks = {
        "all": np.ones_like(data["mixed"], dtype=bool),
        **(
            {
                f"{entry['category']} / fur": data["fur"],
                f"{entry['category']} / hard": ~data["fur"],
            }
            if "fur" in data
            else {}
        ),
    }
    for group, region in masks.items():
        mixed = data["mixed"] & region
        if not region.any():
            continue
        result = collected.setdefault((group, method), Collected())
        geometry = region & (gt_inverse > 0)
        historical_errors = mixed_errors(
            historical_inverse, prediction.alpha[0], reference_inv, reference_alpha, mixed
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
        result.append("edge_composite", reconstruction[edge & region])
        result.append("edge_insert", merge[edge & region])
        result.append("native_predicted_alpha", prediction.alpha[0][mixed])
        result.append("native_target_alpha", target.alpha[0][mixed])
        result.append("predicted_alpha", prediction.alpha[0][mixed])
        result.append("target_alpha", reference_alpha[0][mixed])
        for key, array in {
            "layer_premult_rgb_mae": layer_rgb,
            "sample_count_mae_a003": count_error,
            "depth_order_violation_fraction": reversed_order,
            "composite_mae": reconstruction,
            "insert_z3m_mae": merge,
            "depth_absrel": relative,
            "depth_absrel_scale_aligned": relative_aligned,
            "depth_delta1": delta1,
        }.items():
            selected = geometry if key.startswith("depth_") else region
            result.sums[key] = result.sums.get(key, 0.0) + float(
                array[selected].sum(dtype=np.float64)
            )
        result.pixels += int(region.sum())
        result.geometry_pixels += int(geometry.sum())
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
    export_frames: int = 1,
    baseline_cache: Path | None = None,
    prediction_caches: list[Path] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    entries = entries_for(index, protocol, "test")
    output.mkdir(parents=True, exist_ok=True)
    collected: dict[tuple[str, str], Collected] = {}
    for entry in entries:
        target, data = load_layers(Path(entry["cache"]))
        predictions: dict[str, Layers] = {}
        if baseline_cache:
            predictions = cached_baselines(
                entry["id"], baseline_cache, data["beauty"].astype(np.float32)
            )
        for method, prediction in predictions.items():
            collect(prediction, target, data, method, entry, collected)
    temporal: dict[str, Any] = {}
    for checkpoint in checkpoints:
        model, config = load_model(checkpoint, device)
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
            save_arrays(
                predictions_dir / f"{entry['id']}.npz",
                inverse_depth=prediction.inverse_depth,
                alpha=prediction.alpha,
                rgb=prediction.rgb,
            )
            if i < export_frames:
                write_deep(predictions_dir / f"{entry['id']}.exr", prediction)
        temporal[method] = sequence_jitter(entries, predictions_dir)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    for directory in prediction_caches or []:
        metadata: dict[str, Any] = json.loads((directory / "prediction.json").read_text())
        config = metadata["config"]
        if metadata["test_ids"] != [e["id"] for e in entries]:
            raise ValueError("Prediction cache holdout differs from frozen protocol")
        method = f"{config['preset']} {config['mode']} {config['scale_mode']}"
        fake_method = f"fake {config['preset']} {config['scale_mode']}"
        include_baseline = not any(key[1] == fake_method for key in collected)
        for entry in entries:
            target, data = load_layers(Path(entry["cache"]))
            prediction, _ = load_layers(directory / f"{entry['id']}.npz")
            collect(prediction, target, data, method, entry, collected)
            if include_baseline:
                base_data = np.load(Path(config["features"]) / entry["id"] / "base.npz")
                key = "raw_inverse_depth" if config["scale_mode"] == "raw" else "inverse_depth"
                baseline = fake_deep(base_data[key], data["beauty"].astype(np.float32))
                collect(baseline, target, data, fake_method, entry, collected)
        temporal[method] = sequence_jitter(entries, directory)
    summary = [
        {"group": group, "method": method, **values.summary()}
        for (group, method), values in sorted(collected.items())
    ]
    report = {
        "summary": summary,
        "test_ids": [e["id"] for e in entries],
        "protocol": json.loads(protocol.read_text()),
        "temporal_jitter": temporal,
        "prediction_metadata": {
            str(p): json.loads((p / "prediction.json").read_text()) for p in prediction_caches or []
        },
        "temporal_reason": (
            "Report static-track proxy only when contiguous frames exist; v004 has none."
        ),
        "boundary_protocol": (
            "Directed adjacent-pixel ratios at 5,10,15,20,25%, threshold-proportional "
            "weighting. Exact native-pixel matching, four directed orientations. "
            "Affine-aligned results separate. Native silhouettes use hero instance IDs 2/3; "
            "v004 silhouette proxy is coverage at Z<3m "
            "thresholded at .1; scanline connected-run NMS. Not a published-benchmark "
            "reproduction."
        ),
        "baseline_color_policy": (
            "Analytic/probe lack RGB predictions; reuse observed beauty at every layer. "
            "Composite zero is tautological. Insert-plane error tests geometry but is "
            "not hidden-color recovery."
        ),
        "historical_protocol": (
            "First predicted coverage/depth and last GT-valid ordered depth slot are "
            "compared with original v004 two-layer GT/masks. No visibility compaction "
            "or posthoc depth sorting. Native K metrics remain separate; the historical "
            "probe is oracle-conditioned. protocol.scale documents that legacy track, "
            "not the raw checkpoint inputs."
        ),
        "peak_vram_bytes": torch.cuda.max_memory_allocated()
        if checkpoints and device == "cuda"
        else 0,
    }
    write_json(output / "metrics.json", report)
    return report
