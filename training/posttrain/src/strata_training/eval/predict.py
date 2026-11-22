import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.data.dataset import Frame
from strata_training.data.exr import write_deep
from strata_training.eval.protocol import entries_for
from strata_training.io import save_arrays, write_json
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead
from strata_training.types import Layers


def load_model(path: Path, device: str = "cuda") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    config: dict[str, Any] = checkpoint["config"]
    if config["mode"] == "decoder":
        model: nn.Module = DeepDepthPro(config["preset"], config["layers"], decoder_trainable=False)
    else:
        channels = 32 if config["preset"] == "depthpro2-s" else 128
        model = DeepHead(channels, nn.Identity(), config["layers"])
    missing, unexpected = cast(
        tuple[list[str], list[str]], model.load_state_dict(checkpoint["model"], strict=False)
    )
    if unexpected or any(not k.startswith("base.encoder.") for k in missing):
        raise ValueError(f"Model state mismatch: {missing}, {unexpected}")
    if isinstance(model, DeepDepthPro):
        model.base.decoder.to(device)
        model.base.head.to(device)
        model.deep_head.to(device)
    else:
        model.to(device)
    return model.eval(), config


def predict(model: nn.Module, config: dict[str, Any], frame: Frame) -> Layers:
    height, width = frame.base.shape
    decoder = config["mode"] == "decoder"
    device = (
        next(cast(DeepDepthPro, model).deep_head.parameters()).device
        if decoder
        else next(model.parameters()).device
    )
    core, halo = (128, 192) if decoder else (256, 32)
    full_height, full_width = (
        (height + -height % 32, width + -width % 32) if decoder else (height, width)
    )
    beauty = frame.arrays["beauty"].astype(np.float32)
    beauty = np.pad(
        beauty, ((0, 0), (0, full_height - height), (0, full_width - width)), mode="edge"
    )
    result = Layers(
        np.empty((config["layers"], height, width), dtype=np.float32),
        np.empty((config["layers"], height, width), dtype=np.float32),
        np.empty((config["layers"], 3, height, width), dtype=np.float32),
    )
    arrays = result.arrays()
    for y in range(0, height, core):
        for x in range(0, width, core):
            y0, x0 = max(y - halo, 0), max(x - halo, 0)
            y1, x1 = min(y + core + halo, full_height), min(x + core + halo, full_width)
            image = torch.as_tensor(beauty[:, y0:y1, x0:x1], device=device)[None]
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                if decoder:
                    pyramid = [
                        torch.as_tensor(
                            np.array(
                                p[:, y0 // stride : y1 // stride, x0 // stride : x1 // stride]
                            ),
                            device=device,
                        ).float()[None]
                        for p, stride in zip(frame.pyramid, (4, 8, 16, 32), strict=True)
                    ]
                    scale = torch.tensor(frame.scale, device=device)
                    output = cast(DeepDepthPro, model).from_pyramid(pyramid, image, scale)
                else:
                    features = torch.as_tensor(
                        np.array(frame.features[:, y0:y1, x0:x1]), device=device
                    ).float()[None]
                    base = torch.as_tensor(frame.base[y0:y1, x0:x1], device=device).float()[
                        None, None
                    ]
                    output = cast(dict[str, torch.Tensor], model(features, image, base))
            h, w = min(core, height - y), min(core, width - x)
            for key in arrays:
                value = output[key]
                arrays[key][..., y : y + h, x : x + w] = (
                    value[0, ..., y - y0 : y - y0 + h, x - x0 : x - x0 + w].float().cpu().numpy()
                )
    return result


def cache_predictions(
    index: Path,
    protocol: Path,
    checkpoint: Path,
    output: Path,
    export_frames: int = 1,
    device: str = "cuda",
    scale_mode: str | None = None,
    features: Path | None = None,
) -> dict[str, Any]:
    model, training_config = load_model(checkpoint, device)
    config = dict(training_config)
    if scale_mode is not None:
        if scale_mode not in {"raw", "oracle-aligned"}:
            raise ValueError("Prediction scale must be raw or oracle-aligned")
        config["scale_mode"] = scale_mode
    if features is not None:
        config["features"] = str(features)
    entries = entries_for(index, protocol, "test")
    output.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries):
        frame = Frame(entry, Path(config["features"]), config["scale_mode"])
        prediction = predict(model, config, frame)
        save_arrays(
            output / f"{entry['id']}.npz",
            inverse_depth=prediction.inverse_depth,
            alpha=prediction.alpha,
            rgb=prediction.rgb,
        )
        if i < export_frames:
            write_deep(output / f"{entry['id']}.exr", prediction)
    report = {
        "config": config,
        "training_config": training_config,
        "test_ids": [e["id"] for e in entries],
        "checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
        "device": device,
        "autocast_dtype": "bfloat16",
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
    }
    write_json(output / "prediction.json", report)
    return report
