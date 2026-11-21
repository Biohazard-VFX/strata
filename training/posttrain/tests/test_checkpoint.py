from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from strata_training.model.checkpoint import save_checkpoint
from strata_training.torch_api import api


def _step(model: nn.Linear, optimizer: torch.optim.AdamW, rng: np.random.Generator) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.as_tensor(rng.normal(size=(4, 3)).astype(np.float32))
    loss = model(inputs).square().mean()
    api.autograd.backward(loss)
    api.optim.AdamW.step(optimizer)


def test_checkpoint_continuation_restores_optimizer_and_sampling_state(tmp_path: Path) -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = np.random.default_rng(4004)
    _step(model, optimizer, rng)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, {}, rng, 1, 0.5)
    state: dict[str, Any] = torch.load(path, weights_only=True)
    restored = nn.Linear(3, 1)
    restored.load_state_dict(state["model"])
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = state["rng"]
    _step(model, optimizer, rng)
    _step(restored, resumed_optimizer, resumed_rng)
    torch.testing.assert_close(model.weight, restored.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.bias, restored.bias, rtol=0, atol=0)
