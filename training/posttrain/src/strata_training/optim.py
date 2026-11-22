import math
from typing import Any

import torch
from torch import nn


def parameter_groups(model: nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for name in ("head", "decoder"):
        parameters = [
            p
            for key, p in model.named_parameters()
            if p.requires_grad and (key.startswith("base.") == (name == "decoder"))
        ]
        if parameters:
            rate = config["decoder_learning_rate"] if name == "decoder" else config["learning_rate"]
            groups.append({"name": name, "params": parameters, "lr": rate, "initial_lr": rate})
    return groups


def learning_rate_multiplier(step: int, steps: int, warmup: int) -> float:
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup) / (steps - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def gradient_norm(parameters: list[nn.Parameter]) -> float:
    squares = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(squares).sum().sqrt().item() if squares else 0.0
