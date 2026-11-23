import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.config import training_defaults, validate_config
from strata_training.data.cache import cache_readers
from strata_training.data.dataset import Frame, normalization
from strata_training.data.online import OnlineLoader
from strata_training.data.policy import require_supervision
from strata_training.data.sequences import clip_indices
from strata_training.data.workers import NativeBatches, device_batches
from strata_training.eval.protocol import entries_for
from strata_training.io import write_json
from strata_training.losses.deep import deep_loss
from strata_training.model.checkpoint import save_checkpoint
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead
from strata_training.model.live import FrozenEncoder
from strata_training.optim import gradient_norm, learning_rate_multiplier, parameter_groups
from strata_training.torch_api import api
from strata_training.validation import evaluate_validation, validation_frames


def train(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    indexes = [Path(config["index"])]
    if config.get("validation_index"):
        indexes.append(Path(config["validation_index"]))
    with cache_readers(*indexes):
        return _train(config_path)


def _train(config_path: Path) -> dict[str, Any]:
    config = training_defaults(json.loads(config_path.read_text()))
    config["objective"] = "coverage-bce-logits-v1"
    config["sampling"] = "step-seed-gpu-v2" if config["loader_math"] == "torch" else "step-seed-v2"
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
    require_supervision(entries, purpose="training", root=Path(config["features"]))
    frames = [Frame(e, Path(config["features"])) for e in entries]
    validation = validation_frames(config, entries) if config["early_stopping"] else []
    if validation:
        for name in ("index", "protocol"):
            config[f"validation_{name}_sha256"] = hashlib.sha256(
                Path(config[f"validation_{name}"]).read_bytes()
            ).hexdigest()
    online = (
        OnlineLoader(
            frames,
            Path(config["samples"]),
            config["clean_fraction"],
            config.get("augmentation"),
            config.get("eligibility_warning_fraction", 0.25),
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
    validation_state: dict[str, Any] | None = (
        {"best_metric": None, "best_step": 0, "stale_evals": 0, "history": []}
        if validation
        else None
    )
    if config.get("resume"):
        state: dict[str, Any] = torch.load(config["resume"], map_location="cpu", weights_only=True)
        keys = (
            "preset",
            "mode",
            "layers",
            "index",
            "protocol",
            "features",
            "objective",
            "sampling",
            "learning_rate",
            "decoder_learning_rate",
            "warmup_steps",
            "steps",
            "input_mode",
            "crop",
            "halo",
            "early_stopping",
        )
        if validation:
            keys += (
                "validation_index",
                "validation_protocol",
                "validation_features",
                "validation_index_sha256",
                "validation_protocol_sha256",
                "validation_every",
                "validation_patience",
                "output",
            )
        if config["input_mode"] == "hybrid":
            keys += ("clean_fraction", "samples", "augmentation", "compile_encoder")
        saved_config: dict[str, Any] = {
            "input_mode": "cached",
            "early_stopping": False,
            **state["config"],
        }
        validate_config(saved_config)
        if any(saved_config.get(key) != config.get(key) for key in keys):
            raise ValueError("Resume model/data does not match config")
        missing, unexpected = cast(
            tuple[list[str], list[str]], model.load_state_dict(state["model"], strict=False)
        )
        if unexpected or any(not k.startswith("base.encoder.") for k in missing):
            raise ValueError(f"Resume state mismatch: {missing}, {unexpected}")
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        start_step, previous_seconds = state["step"], state["seconds"]
        if validation:
            validation_state = state.get("validation")
            if validation_state is None:
                raise ValueError("Resume checkpoint is missing validation stopping state")
            if validation_state["stale_evals"] >= config["validation_patience"]:
                raise ValueError("Resume checkpoint already exhausted validation patience")
    start = time.monotonic()
    if start_step >= config["steps"]:
        raise ValueError("Resume step already reaches the requested final step")
    history: list[dict[str, Any]] = []
    if config.get("resume") and (output / "history.json").exists():
        history = json.loads((output / "history.json").read_text())

    def check_validation(step: int) -> dict[str, Any]:
        assert validation_state is not None
        metrics = evaluate_validation(model, config, validation)
        metric = float(metrics["mixed_alpha_mae"])
        improved = (
            validation_state["best_metric"] is None or metric < validation_state["best_metric"]
        )
        validation_state["stale_evals"] = 0 if improved else validation_state["stale_evals"] + 1
        if improved:
            validation_state["best_metric"], validation_state["best_step"] = metric, step
        elapsed = previous_seconds + time.monotonic() - start
        validation_state["history"].append({"step": step, "seconds": elapsed, **metrics})
        write_json(output / "validation.json", validation_state)
        if improved:
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                config,
                rng,
                step,
                elapsed,
                validation=validation_state,
            )
        return metrics

    if validation and not config.get("resume"):
        check_validation(0)
    crop, halo = config["crop"], config["halo"]
    running: list[float] = []
    last_step = start_step
    stopped = False
    with device_batches(NativeBatches(frames, online, clips, config, start_step), device) as stream:
        for step in range(start_step + 1, config["steps"] + 1):
            loading_start = time.monotonic()
            loaded = next(stream)
            selections = loaded.selections
            inputs, target, pyramid, prepared = (
                loaded.inputs,
                loaded.target,
                loaded.pyramid,
                loaded.prepared,
            )
            loader_seconds = time.monotonic() - loading_start
            optimizer.zero_grad(set_to_none=True)
            multiplier = learning_rate_multiplier(
                step, config["steps"], config.get("warmup_steps", 0)
            )
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            model.train()
            with torch.autocast(device, dtype=torch.bfloat16):
                if online is not None and prepared is not None:
                    assert encoder is not None
                    assert teacher is not None
                    prediction = online.predict(
                        prepared, encoder, teacher, model, crop, halo, decoder
                    )
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
            last_step = step
            validation_result: dict[str, Any] | None = None
            if validation and (step % config["validation_every"] == 0 or step == config["steps"]):
                validation_result = check_validation(step)
                assert validation_state is not None
                stopped = validation_state["stale_evals"] >= config["validation_patience"]
            if (
                step % config["log_every"] == 0
                or step == config["steps"]
                or validation_result is not None
            ):
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
                    "validation": validation_result,
                }
                history.append(row)
                running.clear()
                write_json(output / "history.json", history)
                save_checkpoint(
                    output / "last.pt",
                    model,
                    optimizer,
                    config,
                    rng,
                    step,
                    row["seconds"],
                    validation=validation_state,
                )
            if stopped:
                break
    seconds = previous_seconds + time.monotonic() - start
    save_checkpoint(
        output / "final.pt",
        model,
        optimizer,
        config,
        rng,
        last_step,
        seconds,
        validation=validation_state,
    )
    report = {
        "config": config,
        "seconds": seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "train_frames": len(frames),
        "recomposition_eligible_fraction": online.eligible_fraction if online else None,
        "final_train_loss": history[-1]["train_loss"],
        "stopped_step": last_step,
        "stopping_reason": "validation_patience" if stopped else "step_ceiling",
        "best_step": validation_state["best_step"] if validation_state else last_step,
        "best_validation_mixed_alpha_mae": validation_state["best_metric"]
        if validation_state
        else None,
        "selected_checkpoint": str(output / ("best.pt" if validation else "final.pt")),
    }
    write_json(output / "training.json", report)
    return report
