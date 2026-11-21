import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.data.dataset import Frame
from strata_training.data.exr import write_deep
from strata_training.device import configure_gpu
from strata_training.eval.protocol import entries_for
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead
from strata_training.types import Layers


def load_model(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
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
        model.base.decoder.cuda()
        model.base.head.cuda()
        model.deep_head.cuda()
    else:
        model.cuda()
    return model.eval(), config


def predict(model: nn.Module, config: dict[str, Any], frame: Frame) -> Layers:
    height, width = frame.base.shape
    decoder = config["mode"] == "decoder"
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
            image = torch.as_tensor(beauty[:, y0:y1, x0:x1], device="cuda")[None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if decoder:
                    pyramid = [
                        torch.as_tensor(
                            np.array(
                                p[:, y0 // stride : y1 // stride, x0 // stride : x1 // stride]
                            ),
                            device="cuda",
                        ).float()[None]
                        for p, stride in zip(frame.pyramid, (4, 8, 16, 32), strict=True)
                    ]
                    scale = torch.tensor(frame.scale, device="cuda")
                    output = cast(DeepDepthPro, model).from_pyramid(pyramid, image, scale)
                else:
                    features = torch.as_tensor(
                        np.array(frame.features[:, y0:y1, x0:x1]), device="cuda"
                    ).float()[None]
                    base = torch.as_tensor(frame.base[y0:y1, x0:x1], device="cuda").float()[
                        None, None
                    ]
                    output = cast(dict[str, torch.Tensor], model(features, image, base))
            h, w = min(core, height - y), min(core, width - x)
            for key, value in output.items():
                arrays[key][..., y : y + h, x : x + w] = (
                    value[0, ..., y - y0 : y - y0 + h, x - x0 : x - x0 + w].float().cpu().numpy()
                )
    return result


def cache_predictions(
    index: Path, protocol: Path, checkpoint: Path, output: Path, export_frames: int = 1
) -> dict[str, Any]:
    configure_gpu()
    model, config = load_model(checkpoint)
    entries = entries_for(index, protocol, "test")
    output.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries):
        frame = Frame(entry, Path(config["features"]), config["scale_mode"])
        prediction = predict(model, config, frame)
        np.savez_compressed(
            output / f"{entry['id']}.npz",
            inverse_depth=prediction.inverse_depth,
            alpha=prediction.alpha,
            rgb=prediction.rgb,
        )
        if i < export_frames:
            write_deep(output / f"{entry['id']}.exr", prediction)
        print(json.dumps({"prediction": entry["id"]}), flush=True)
    report = {
        "config": config,
        "test_ids": [e["id"] for e in entries],
        "checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (output / "prediction.json").write_text(json.dumps(report, indent=2))
    return report
