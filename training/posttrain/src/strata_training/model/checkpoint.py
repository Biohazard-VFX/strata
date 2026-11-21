from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.model.dp2 import DeepDepthPro


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    rng: np.random.Generator,
    step: int,
    seconds: float,
) -> None:
    state = {
        "model": {k: v for k, v in model.state_dict().items() if not k.startswith("base.encoder.")},
        "optimizer": optimizer.state_dict(),
        "config": config,
        "rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "step": step,
        "seconds": seconds,
    }
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def load_online(path: Path, device: str = "cpu") -> tuple[DeepDepthPro, dict[str, Any]]:
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    config: dict[str, Any] = state["config"]
    model = DeepDepthPro(config["preset"], config["layers"], decoder_trainable=False)
    if config["mode"] == "head":
        missing, unexpected = cast(
            tuple[list[str], list[str]],
            model.deep_head.load_state_dict(state["model"], strict=False),
        )
        allowed = "inverse_head."
    elif config["mode"] == "decoder":
        missing, unexpected = cast(
            tuple[list[str], list[str]], model.load_state_dict(state["model"], strict=False)
        )
        allowed = "base.encoder."
    else:
        raise ValueError("Unsupported checkpoint mode")
    if unexpected or any(not k.startswith(allowed) for k in missing):
        raise ValueError(f"Online state mismatch: {missing}, {unexpected}")
    return model.to(device).eval().requires_grad_(False), config
