"""Keep the final inverse-depth projection in float32 while large activations stay fp16."""

from typing import cast

import torch
from torch import Tensor, nn


class FloatProjection(nn.Module):
    def __init__(self, projection: nn.Module) -> None:
        super().__init__()
        self.projection = projection

    def forward(self, image: Tensor) -> Tensor:
        with torch.autocast(image.device.type, enabled=False):
            return cast("Tensor", self.projection(image.float()))
