import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from strata_training.data.cache import display_rgb
from strata_training.data.metadata import camera_frames
from strata_training.data.online import image_digest
from strata_training.data.scale import oracle_metric_scale
from strata_training.data.sequences import clip_indices
from strata_training.io import save_array, save_arrays, write_json
from strata_training.model.dp2 import DeepDepthPro
from strata_training.types import Array


def _before(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    del inputs
    module.cuda()


def _after(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    del inputs, output
    module.cpu()


def _load(preset: str, device: str) -> DeepDepthPro:
    model = DeepDepthPro(preset, decoder_trainable=False)
    if preset == "depthpro2-h" and device == "cuda":
        blocks = model.base.encoder.blocks
        model.base.encoder.blocks = nn.ModuleList()
        model.cuda()
        model.base.encoder.blocks = blocks
        for block in blocks:
            block.register_forward_pre_hook(_before)
            block.register_forward_hook(_after)
    else:
        model.to(device)
    return model.eval()


def _save(
    entry: dict[str, Any],
    root: Path,
    pyramid: list[torch.Tensor],
    features: torch.Tensor,
    focal_inv: Array,
    clip_ids: list[str],
) -> None:
    target = root / entry["id"]
    target.mkdir(exist_ok=True)
    data = np.load(entry["cache"])
    metric_scale = oracle_metric_scale(data, focal_inv)
    for level, tensor in enumerate(pyramid):
        save_array(target / f"pyramid_{level}.npy", tensor.half().cpu().numpy())
    save_array(target / "features.npy", features.half().cpu().numpy())
    camera = camera_frames(Path(entry["manifest"]).parent)[entry["frame"]]
    raw_scale = float(camera["width"]) / float(camera["fx"])
    write_json(target / "context.json", {"clip_ids": clip_ids})
    save_arrays(
        target / "base.npz",
        inverse_depth=focal_inv * metric_scale,
        metric_scale=metric_scale,
        raw_inverse_depth=focal_inv * raw_scale,
        raw_metric_scale=raw_scale,
    )


def _denoised(entry: dict[str, Any], samples: Path | None) -> Array | None:
    if samples is not None:
        return np.load(samples / entry["id"] / "denoised.npy")
    with np.load(entry["cache"]) as data:
        return data["denoised"].astype(np.float32) if "denoised" in data else None


def extract(
    index: Path,
    output: Path,
    preset: str,
    clip_length: int = 1,
    device: str = "cuda",
    samples: Path | None = None,
) -> None:
    entries: list[dict[str, Any]] = json.loads(index.read_text())
    if clip_length < 1:
        raise ValueError("clip_length must be positive")
    if samples is not None and clip_length != 1:
        raise ValueError("Denoised clean caches require independent single frames")
    by_shot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_shot[entry["shot"]].append(entry)
    model = _load(preset, device)
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    start = time.monotonic()
    completed = 0
    for shot in by_shot.values():
        for indices in clip_indices(shot, clip_length):
            clip = [shot[i] for i in indices]
            ids = [e["id"] for e in clip]
            denoises = [_denoised(e, samples) for e in clip]
            if all((output / e["id"] / "base.npz").exists() for e in clip):
                for entry, denoise in zip(clip, denoises, strict=True):
                    if denoise is not None:
                        context = json.loads((output / entry["id"] / "context.json").read_text())
                        if context.get("image_sha256") != image_digest(denoise):
                            raise ValueError(
                                "Existing clean cache input changed; use a new directory"
                            )
                completed += len(clip)
                continue
            arrays = [
                np.load(e["cache"])["srgb"] if denoise is None else display_rgb(denoise, "ACEScg")
                for e, denoise in zip(clip, denoises, strict=True)
            ]
            image = torch.as_tensor(np.stack(arrays).astype(np.float32) / 255, device=device)
            height, width = image.shape[-2:]
            normalized = torch.nn.functional.pad(
                (image - mean) / std, (0, -width % 32, 0, -height % 32), mode="replicate"
            )
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False):
                pyramid = model.pyramid(normalized)
                for i, entry in enumerate(clip):
                    frame_pyramid = [p[i : i + 1] for p in pyramid]
                    features = model.decode(frame_pyramid)
                    with torch.autocast(device, enabled=False):
                        inverse = (
                            model.deep_head.inverse_head(features.float())[0, 0, :height, :width]
                            .float()
                            .cpu()
                            .numpy()
                        )
                    full_features = features[0, :, :height, :width]
                    _save(
                        entry,
                        output,
                        [p[0] for p in frame_pyramid],
                        full_features,
                        inverse,
                        ids,
                    )
                    completed += 1
                    denoise = denoises[i]
                    if denoise is not None:
                        save_array(
                            output / entry["id"] / "image.npy",
                            denoise.astype(np.float32),
                        )
                        write_json(
                            output / entry["id"] / "context.json",
                            {
                                "clip_ids": ids,
                                "image_sha256": image_digest(denoise),
                            },
                        )
    write_json(
        output / "extraction.json",
        {
            "preset": preset,
            "clip_length": clip_length,
            "seconds": time.monotonic() - start,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "frames": completed,
            "encoder_offload": preset == "depthpro2-h",
        },
    )
