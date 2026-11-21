import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from strata_training.data.sequences import clip_indices
from strata_training.device import configure_gpu
from strata_training.model.dp2 import DeepDepthPro
from strata_training.types import Array


def _before(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    del inputs
    module.cuda()


def _after(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    del inputs, output
    module.cpu()


def _load(preset: str) -> DeepDepthPro:
    model = DeepDepthPro(preset, decoder_trainable=False)
    if preset == "depthpro2-h":
        model.base.encoder.bfloat16()
        blocks = model.base.encoder.blocks
        model.base.encoder.blocks = nn.ModuleList()
        model.cuda()
        model.base.encoder.blocks = blocks
        for block in blocks:
            block.register_forward_pre_hook(_before)
            block.register_forward_hook(_after)
    else:
        model.cuda()
    return model.eval()


def _save(
    entry: dict[str, Any],
    root: Path,
    pyramid: list[torch.Tensor],
    features: torch.Tensor | None,
    focal_inv: Array,
    clip_ids: list[str],
) -> None:
    target = root / entry["id"]
    target.mkdir(exist_ok=True)
    data = np.load(entry["cache"])
    legacy = Path(entry["legacy_data"]) if entry["legacy_data"] else None
    if legacy:
        old = np.load(legacy / "gt.npz")
        solid = ndimage.binary_erosion(
            (~old["two"]) & (old["A_tot"] > 0.999) & (old["inv_blend"] > 0), iterations=4
        )
        truth = old["inv_blend"]
    else:
        solid = (
            (data["alpha"][0] > 0.999)
            & (data["alpha"][1:].sum(axis=0) < 1e-6)
            & (data["inverse_depth"][0] > 0)
        )
        truth = data["inverse_depth"][0]
    scale = float(np.exp(np.median(np.log(np.maximum(focal_inv[solid], 1e-8) / truth[solid]))))
    for level, tensor in enumerate(pyramid):
        np.save(target / f"pyramid_{level}.npy", tensor.half().cpu().numpy())
    if features is None:
        if legacy is None:
            raise ValueError("Reusing features requires a legacy source")
        if not (target / "features.npy").exists():
            (target / "features.npy").symlink_to(legacy / "features.npy")
    else:
        np.save(target / "features.npy", features.half().cpu().numpy())
    manifest = json.loads(Path(entry["manifest"]).read_text())
    camera = json.loads((Path(entry["manifest"]).parent / "camera.json").read_text())
    focal = float(camera["frames"][f"{entry['frame']:04d}"]["intrinsics"][0][0])
    raw_scale = float(manifest["width"]) / focal
    (target / "context.json").write_text(json.dumps({"clip_ids": clip_ids}))
    np.savez_compressed(
        target / "base.npz",
        inverse_depth=focal_inv / scale,
        metric_scale=1 / scale,
        raw_inverse_depth=focal_inv * raw_scale,
        raw_metric_scale=raw_scale,
    )


def extract(index: Path, output: Path, preset: str, clip_length: int = 1) -> None:
    entries: list[dict[str, Any]] = json.loads(index.read_text())
    if clip_length < 1:
        raise ValueError("clip_length must be positive")
    by_shot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_shot[entry["shot"]].append(entry)
    configure_gpu()
    model = _load(preset)
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    start = time.monotonic()
    completed = 0
    for shot in by_shot.values():
        for indices in clip_indices(shot, clip_length):
            clip = [shot[i] for i in indices]
            ids = [e["id"] for e in clip]
            if all((output / e["id"] / "base.npz").exists() for e in clip):
                completed += len(clip)
                continue
            arrays = [np.load(e["cache"])["srgb"] for e in clip]
            image = torch.as_tensor(np.stack(arrays).astype(np.float32) / 255, device="cuda")
            height, width = image.shape[-2:]
            normalized = torch.nn.functional.pad(
                (image - mean) / std, (0, -width % 32, 0, -height % 32), mode="replicate"
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                pyramid = model.pyramid(normalized)
                for i, entry in enumerate(clip):
                    frame_pyramid = [p[i : i + 1] for p in pyramid]
                    if preset == "depthpro2-h" and entry["legacy_data"] and len(clip) == 1:
                        inverse = np.load(Path(entry["legacy_data"]) / "dp2.npz")["inverse_depth"]
                        full_features = None
                    else:
                        features = model.decode(frame_pyramid)
                        with torch.autocast("cuda", enabled=False):
                            inverse = (
                                model.deep_head.inverse_head(features.float())[
                                    0, 0, :height, :width
                                ]
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
                    print(
                        json.dumps(
                            {"id": entry["id"], "clip": ids, "seconds": time.monotonic() - start}
                        ),
                        flush=True,
                    )
    (output / "extraction.json").write_text(
        json.dumps(
            {
                "preset": preset,
                "clip_length": clip_length,
                "seconds": time.monotonic() - start,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "frames": completed,
                "encoder_offload": preset == "depthpro2-h",
            },
            indent=2,
        )
    )
