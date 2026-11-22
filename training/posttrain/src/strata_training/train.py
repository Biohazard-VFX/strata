import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.config import training_defaults, validate_config
from strata_training.data.dataset import Frame, batch, normalization
from strata_training.data.online import OnlineLoader
from strata_training.data.sequences import clip_indices
from strata_training.eval.protocol import entries_for
from strata_training.io import write_json
from strata_training.losses.deep import deep_loss
from strata_training.model.checkpoint import save_checkpoint
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead
from strata_training.model.live import FrozenEncoder
from strata_training.optim import gradient_norm, learning_rate_multiplier, parameter_groups
from strata_training.torch_api import api


def train(config_path: Path) -> dict[str, Any]:
    config = training_defaults(json.loads(config_path.read_text()))
    config["objective"] = "coverage-bce-logits-v1"
    validate_config(config)
    device = config.get("device", "cuda")
    api.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    entries = entries_for(Path(config["index"]), Path(config["protocol"]), "train")
    if any(e.get("compact_scene_only", False) for e in entries):
        raise ValueError(
            "Compact datagen targets omit sky colour/coverage; background supervision "
            "must be published before training this representation"
        )
    frames = [Frame(e, Path(config["features"]), config["scale_mode"]) for e in entries]
    online = (
        OnlineLoader(
            frames, Path(config["samples"]), config["clean_fraction"], config.get("augmentation")
        )
        if config["input_mode"] == "hybrid"
        else None
    )
    if any(min(frame.base.shape) < config["crop"] + 2 * config["halo"] for frame in frames):
        raise ValueError("Native frames must fit the requested crop plus halo without resizing")
    clip_length = int(config.get("clip_length", 1))
    clips = clip_indices(entries, clip_length)
    if clip_length > 1 and not any(len(c) > 1 for c in clips):
        raise ValueError("Requested temporal training but dataset contains no contiguous sequences")
    decoder = config["mode"] == "decoder"
    if decoder:
        network = DeepDepthPro(config["preset"], config["layers"], decoder_trainable=True)
        network.base.decoder.to(device)
        network.base.head.to(device)
        network.deep_head.to(device)
        head = network.deep_head
        model: nn.Module = network
    else:
        channels = frames[0].features.shape[0]
        head = DeepHead(channels, nn.Identity(), config["layers"]).to(device)
        model = head
    teacher: DeepDepthPro | None = None
    encoder: FrozenEncoder | None = None
    if online is not None:
        teacher = (
            cast(DeepDepthPro, model)
            if decoder
            else DeepDepthPro(config["preset"], config["layers"], decoder_trainable=False)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        teacher.base.encoder.to(device, dtype=torch.bfloat16)
        encoder = FrozenEncoder(teacher.base.encoder, compiled=config["compile_encoder"])
    mean, std = normalization(frames)
    head.input_mean.copy_(torch.as_tensor(mean).to(device))
    head.input_std.copy_(torch.as_tensor(std).to(device))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameter_groups(model, config), weight_decay=1e-4)
    start_step = 0
    previous_seconds = 0.0
    if config.get("resume"):
        state: dict[str, Any] = torch.load(config["resume"], map_location="cpu", weights_only=True)
        keys = (
            "preset",
            "mode",
            "layers",
            "scale_mode",
            "index",
            "protocol",
            "features",
            "objective",
            "learning_rate",
            "decoder_learning_rate",
            "warmup_steps",
            "steps",
            "input_mode",
            "crop",
            "halo",
        )
        if config["input_mode"] == "hybrid":
            keys += ("clean_fraction", "samples", "augmentation", "compile_encoder")
        saved_config = {"input_mode": "cached", **state["config"]}
        if any(saved_config.get(key) != config.get(key) for key in keys):
            raise ValueError("Resume model/data/scale does not match config")
        missing, unexpected = cast(
            tuple[list[str], list[str]], model.load_state_dict(state["model"], strict=False)
        )
        if unexpected or any(not k.startswith("base.encoder.") for k in missing):
            raise ValueError(f"Resume state mismatch: {missing}, {unexpected}")
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        start_step, previous_seconds = state["step"], state["seconds"]
    start = time.monotonic()
    if start_step >= config["steps"]:
        raise ValueError("Resume step already reaches the requested final step")
    history: list[dict[str, Any]] = []
    if config.get("resume") and (output / "history.json").exists():
        history = json.loads((output / "history.json").read_text())
    crop, halo = config["crop"], config["halo"]
    running: list[float] = []
    for step in range(start_step + 1, config["steps"] + 1):
        selections: list[tuple[int, int, int]] = []
        while len(selections) < config["batch_size"]:
            clip = clips[int(rng.integers(len(clips)))]
            coordinates = frames[clip[0]].crop(rng, crop + 2 * halo)
            selections.extend(
                (i, *coordinates) for i in clip[: config["batch_size"] - len(selections)]
            )
        inputs: dict[str, torch.Tensor] = {}
        pyramid: list[torch.Tensor] = []
        loading_start = time.monotonic()
        prepared = online.prepare(selections, rng, crop, halo, device) if online else None
        if prepared is None:
            inputs, target, pyramid = batch(
                frames, selections, crop, halo, device=device, decoder=decoder
            )
        else:
            target = prepared.target
        loader_seconds = time.monotonic() - loading_start
        optimizer.zero_grad(set_to_none=True)
        multiplier = learning_rate_multiplier(step, config["steps"], config.get("warmup_steps", 0))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        model.train()
        with torch.autocast(device, dtype=torch.bfloat16):
            if online is not None and prepared is not None:
                assert encoder is not None
                assert teacher is not None
                prediction = online.predict(prepared, encoder, teacher, model, crop, halo, decoder)
            elif decoder:
                prediction = cast(DeepDepthPro, model).from_pyramid(
                    pyramid, inputs["image"], inputs["scale"]
                )
            else:
                prediction = cast(
                    dict[str, torch.Tensor],
                    model(inputs["features"], inputs["image"], inputs["base"]),
                )
            central = {
                k: v[..., halo : halo + crop, halo : halo + crop] for k, v in prediction.items()
            }
            loss, terms = deep_loss(central, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        api.autograd.backward(loss)
        norms = {str(g["name"]): gradient_norm(g["params"]) for g in optimizer.param_groups}
        if not all(np.isfinite(value) for value in norms.values()):
            raise FloatingPointError(f"Nonfinite gradients at step {step}")
        torch.nn.utils.clip_grad_norm_(parameters, 5)
        api.optim.AdamW.step(optimizer)
        running.append(loss.item())
        if step % config["log_every"] == 0 or step == config["steps"]:
            row: dict[str, Any] = {
                "step": step,
                "seconds": previous_seconds + time.monotonic() - start,
                "train_loss": float(np.mean(running)),
                "last_terms": {k: v.item() for k, v in terms.items()},
                "gradient_norms": norms,
                "learning_rates": {str(g["name"]): g["lr"] for g in optimizer.param_groups},
                "alpha_logit_abs_max": central["alpha_logits"].detach().abs().max().item(),
                "alpha_mae": (central["alpha"].detach() - target["alpha"]).abs().mean().item(),
                "mixed_alpha_mae": (
                    (central["alpha"][:, 0].detach() - target["alpha"][:, 0])
                    .abs()[target["mixed"].bool()]
                    .mean()
                    .item()
                    if target["mixed"].bool().any().item()
                    else None
                ),
                "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                "loader_seconds": loader_seconds,
                "clean_samples": len(prepared.clean) if prepared else len(selections),
                "live_samples": len(prepared.live) if prepared else 0,
            }
            history.append(row)
            running.clear()
            write_json(output / "history.json", history)
            save_checkpoint(output / "last.pt", model, optimizer, config, rng, step, row["seconds"])
    seconds = previous_seconds + time.monotonic() - start
    save_checkpoint(output / "final.pt", model, optimizer, config, rng, config["steps"], seconds)
    report = {
        "config": config,
        "seconds": seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "train_frames": len(frames),
        "final_train_loss": history[-1]["train_loss"],
    }
    write_json(output / "training.json", report)
    return report
