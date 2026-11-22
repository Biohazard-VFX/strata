import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.nn import functional as functional

from strata_training.baselines.curves import apply_curve
from strata_training.baselines.deep import analytic
from strata_training.eval.predict import load_model
from strata_training.eval.real.fetch import checksum
from strata_training.eval.real.protocol import deep_to_matte
from strata_training.eval.real.readers import samples
from strata_training.eval.real.runner import cache_path
from strata_training.io import save_arrays, write_json
from strata_training.model.dp2 import DeepDepthPro


def infer(
    root: Path,
    output: Path,
    checkpoint: Path,
    curves: Path,
    limit: int,
    max_side: int,
) -> dict[str, Any]:
    if limit < 1 or max_side < 32:
        raise ValueError("Positive limit and max_side>=32 are required")
    head, config = load_model(checkpoint, "cpu")
    if config["mode"] != "head" or config["preset"] != "depthpro2-h":
        raise ValueError("Expected the frozen H+ head-only checkpoint")
    encoder = DeepDepthPro(config["preset"], decoder_trainable=False).eval()
    curve = np.asarray(json.loads(curves.read_text())["curves"]["16"])
    rows: list[dict[str, Any]] = []
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    for sample in samples(root):
        image = torch.as_tensor(sample.image.transpose(2, 0, 1).copy())[None]
        height, width = image.shape[-2:]
        scale = min(1.0, max_side / max(height, width))
        resized = functional.interpolate(
            image,
            size=(max(32, int(height * scale)), max(32, int(width * scale))),
            mode="bilinear",
            align_corners=False,
        )
        h, w = resized.shape[-2:]
        normalized = functional.pad(
            (resized - mean) / std, (0, -w % 32, 0, -h % 32), mode="replicate"
        )
        linear = torch.where(
            resized <= 0.04045, resized / 12.92, ((resized + 0.055) / 1.055) ** 2.4
        )
        with torch.inference_mode():
            pyramid = encoder.pyramid(normalized)
            features = encoder.decode(pyramid)[..., :h, :w].float()
            inverse = encoder.deep_head.inverse_head(features).clamp_min(1e-5)
            result = cast(dict[str, torch.Tensor], head(features, linear, inverse))
            alpha = deep_to_matte(result["alpha"][0].float().numpy())
            raw, _, _ = analytic(inverse[0, 0].float().numpy(), 16)
            corrected = apply_curve(raw, curve)
            for name, matte in (("h-head", alpha), ("analytic-r16-curve", corrected)):
                native = functional.interpolate(
                    torch.as_tensor(matte.copy())[None, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                save_arrays(cache_path(output / name, sample.identifier), alpha=native)
        rows.append(
            {
                "id": sample.identifier,
                "input_height": h,
                "input_width": w,
                "native_height": height,
                "native_width": width,
            }
        )
        if len(rows) >= limit:
            break
    report = {
        "device": "cpu",
        "track": "resized-input/native-label diagnostic",
        "metric_scale": 1,
        "metric_scale_note": "fixed unit-focal; no real camera calibration",
        "checkpoint_sha256": checksum(checkpoint),
        "curves_sha256": checksum(curves),
        "max_side": max_side,
        "limit": limit,
        "samples": rows,
    }
    write_json(output / "inference.json", report)
    return report
