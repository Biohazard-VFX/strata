import json
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.data.dataset import Frame, batch, normalization
from strata_training.device import configure_gpu
from strata_training.eval.protocol import entries_for
from strata_training.losses.deep import deep_loss
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead
from strata_training.torch_api import api


def train(config_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(config_path.read_text())
    configure_gpu()
    api.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    entries = entries_for(Path(config["index"]), Path(config["protocol"]), "train")
    frames = [Frame(e, Path(config["features"]), config["scale_mode"]) for e in entries]
    decoder = config["mode"] == "decoder"
    if decoder:
        network = DeepDepthPro(config["preset"], config["layers"], decoder_trainable=True)
        network.base.decoder.cuda()
        network.base.head.cuda()
        network.deep_head.cuda()
        head = network.deep_head
        model: nn.Module = network
    else:
        channels = frames[0].features.shape[0]
        head = DeepHead(channels, nn.Identity(), config["layers"]).cuda()
        model = head
    mean, std = normalization(frames)
    head.input_mean.copy_(torch.as_tensor(mean).cuda())
    head.input_std.copy_(torch.as_tensor(std).cuda())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=1e-4)
    start_step = 0
    previous_seconds = 0.0
    if config.get("resume"):
        state: dict[str, Any] = torch.load(
            config["resume"], map_location="cuda", weights_only=False
        )
        if (
            state["config"]["preset"] != config["preset"]
            or state["config"]["mode"] != config["mode"]
        ):
            raise ValueError("Resume model does not match config")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        start_step, previous_seconds = state["step"], state["seconds"]
    start = time.monotonic()
    history: list[dict[str, Any]] = []
    crop, halo = config["crop"], config["halo"]
    running: list[float] = []
    for step in range(start_step + 1, config["steps"] + 1):
        selections: list[tuple[int, int, int]] = []
        for _ in range(config["batch_size"]):
            i = int(rng.integers(len(frames)))
            selections.append((i, *frames[i].crop(rng, crop + 2 * halo)))
        inputs, target, pyramid = batch(frames, selections, crop, halo, decoder=decoder)
        optimizer.zero_grad(set_to_none=True)
        learning_rate = config["learning_rate"] * (
            0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / config["steps"]))
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if decoder:
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
        torch.nn.utils.clip_grad_norm_(parameters, 5)
        api.optim.AdamW.step(optimizer)
        running.append(loss.item())
        if step % config["log_every"] == 0 or step == config["steps"]:
            row = {
                "step": step,
                "seconds": previous_seconds + time.monotonic() - start,
                "train_loss": float(np.mean(running)),
                "last_terms": {k: v.item() for k, v in terms.items()},
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
            history.append(row)
            running.clear()
            (output / "history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)
    seconds = previous_seconds + time.monotonic() - start
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "step": config["steps"],
            "seconds": seconds,
        },
        output / "final.pt",
    )
    report = {
        "config": config,
        "seconds": seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "train_frames": len(frames),
        "final_train_loss": history[-1]["train_loss"],
    }
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report
