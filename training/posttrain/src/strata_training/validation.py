from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from strata_training.data.cache import load_layers
from strata_training.data.dataset import Frame
from strata_training.data.policy import require_supervision
from strata_training.data.targets import composite
from strata_training.eval.coverage import coverage_samples, coverage_summary
from strata_training.eval.leakage import leakage_samples, leakage_summary
from strata_training.eval.predict import predict
from strata_training.eval.protocol import entries_for
from strata_training.losses.deep import deep_loss
from strata_training.types import Array


def validation_frames(config: dict[str, Any], training: list[dict[str, Any]]) -> list[Frame]:
    """Own validation arrays/maps; callers serialize evaluation of shared model weights."""
    keys = ("validation_index", "validation_protocol", "validation_features")
    if any(not config.get(key) for key in keys):
        raise ValueError(
            "Early stopping requires an explicit OOD validation index/protocol/features"
        )
    if Path(config["validation_index"]).resolve() == Path(config["index"]).resolve():
        raise ValueError("OOD validation requires a separate scene-distribution index")
    entries = entries_for(
        Path(config["validation_index"]),
        Path(config["validation_protocol"]),
        "test",
        purpose="validation",
    )
    if not entries:
        raise ValueError("OOD validation must contain frames")
    for key in ("id", "shot", "cache", "manifest"):
        seen = {str(e[key]) for e in training if e.get(key) is not None}
        if any(str(e[key]) in seen for e in entries if e.get(key) is not None):
            raise ValueError(
                "OOD validation must be disjoint from training shots, frames and targets"
            )
    require_supervision(entries, purpose="validation", root=Path(config["validation_features"]))
    frames = [Frame(e, Path(config["validation_features"])) for e in entries]
    if not any(frame.arrays["mixed"].any() for frame in frames):
        raise ValueError("OOD validation requires mixed pixels for checkpoint selection")
    return frames


def evaluate_validation(
    model: nn.Module, config: dict[str, Any], frames: list[Frame]
) -> dict[str, Any]:
    """Borrow frames/weights, restore model mode, return pooled alpha MAE and mean frame loss.

    Calls change model mode temporarily and must not overlap training or other evaluations.
    Validation loss reconstructs logits from clipped inference probabilities on CPU.
    """
    require_supervision([frame.entry for frame in frames], purpose="validation")
    was_training = model.training
    error, pixels = 0.0, 0
    losses: list[float] = []
    leakage: dict[str, list[Array]] = {}
    sweep: dict[str, list[Array]] = {}
    model.eval()
    try:
        for frame in frames:
            result = predict(model, config, frame)
            target_layers, arrays = load_layers(Path(frame.entry["cache"]))
            mixed = arrays["mixed"].astype(bool)
            error += float(
                np.abs(result.alpha[0] - arrays["alpha"][0])[mixed].sum(dtype=np.float64)
            )
            pixels += int(mixed.sum())
            prediction = {k: torch.as_tensor(v[None]).float() for k, v in result.arrays().items()}
            prediction["alpha_logits"] = torch.logit(prediction["alpha"].clamp(1e-7, 1 - 1e-7))
            prediction["layer_rgb"] = prediction["rgb"]
            for name, values in leakage_samples(
                result, target_layers, mixed, arrays["beauty"]
            ).items():
                leakage.setdefault(name, []).append(values)
            for name, values in coverage_samples(result, target_layers, mixed).items():
                sweep.setdefault(name, []).append(values)
            target = {
                k: torch.tensor(arrays[k][None]).float()
                for k in ("inverse_depth", "alpha", "rgb", "mixed")
            }
            target["beauty"] = composite(target["alpha"].movedim(1, 0), target["rgb"].movedim(1, 0))
            loss, _ = deep_loss(prediction, target)
            losses.append(loss.item())
    finally:
        model.train(was_training)
    if not pixels:
        raise ValueError("OOD validation requires mixed pixels for checkpoint selection")
    metric, loss = error / pixels, float(np.mean(losses))
    if not np.isfinite(metric) or not np.isfinite(loss):
        raise FloatingPointError("Nonfinite OOD validation metric or loss")
    leak = leakage_summary({name: np.concatenate(values) for name, values in leakage.items()})
    if leak["worst_mean"] is None:
        raise ValueError("OOD validation requires GT-opaque interior pixels for leakage")
    if not np.isfinite(leak["worst_mean"]):
        raise FloatingPointError("Nonfinite OOD leakage")
    sweep_result = coverage_summary(
        {name: np.concatenate(values) for name, values in sweep.items()}
    )
    if not np.isfinite(sweep_result["selection"]):
        raise FloatingPointError("Nonfinite OOD coverage sweep")
    return {
        "coverage_sweep": sweep_result,
        "mixed_alpha_mae": metric,
        "leakage": leak,
        "loss": loss,
        "mixed_pixels": pixels,
        "frames": len(frames),
    }
