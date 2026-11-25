from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from strata_training.io import atomic_path


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    rng: np.random.Generator,
    step: int,
    seconds: float,
    validation: dict[str, Any] | None = None,
) -> None:
    state = {
        "model": {k: v for k, v in model.state_dict().items() if not k.startswith("base.encoder.")},
        "optimizer": optimizer.state_dict(),
        "config": config,
        "rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "step": step,
        "seconds": seconds,
        "validation": validation,
    }
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        torch.save(state, stream)
