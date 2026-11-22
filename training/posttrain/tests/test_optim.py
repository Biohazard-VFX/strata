import pytest
import torch
from torch import nn

from strata_training.config import validate_config
from strata_training.optim import gradient_norm, learning_rate_multiplier, parameter_groups
from strata_training.torch_api import api


def test_pretrained_and_new_parameters_keep_separate_warmup_rates() -> None:
    model = nn.ModuleDict({"base": nn.Linear(2, 2), "deep_head": nn.Linear(2, 1)})
    config = {
        "mode": "decoder",
        "learning_rate": 4e-4,
        "decoder_learning_rate": 5e-6,
        "warmup_steps": 20,
        "steps": 400,
        "batch_size": 1,
        "crop": 32,
        "halo": 0,
    }
    validate_config(config)
    optimizer = torch.optim.AdamW(parameter_groups(model, config))
    assert {g["name"] for g in optimizer.param_groups} == {"head", "decoder"}
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))
    loss = model["deep_head"](model["base"](torch.ones(3, 2))).square().mean()
    api.autograd.backward(loss)
    for group in optimizer.param_groups:
        assert gradient_norm(group["params"]) > 0
        group["lr"] = group["initial_lr"] * learning_rate_multiplier(1, 400, 20)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2.5e-7)
    assert learning_rate_multiplier(20, 400, 20) == 1
    assert learning_rate_multiplier(400, 400, 20) == 0.01
    config["warmup_steps"] = 400
    with pytest.raises(ValueError, match="Warmup"):
        validate_config(config)
