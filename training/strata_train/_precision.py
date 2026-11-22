"""FP16 policy: float32 for the projection and wherever the last tap's massive activations go."""

from typing import cast

import torch
from depth_pro_2 import DepthPro2
from torch import Tensor, nn


class Float32(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, image: Tensor) -> Tensor:
        with torch.autocast(image.device.type, enabled=False):
            return cast("Tensor", self.module(image.float()))


def keep_float32_boundaries(model: DepthPro2) -> None:
    reassemble = model.encoder.reassemble
    for stage in list(reassemble)[:-1]:
        sequential = cast("nn.Sequential", stage)
        sequential[0] = Float32(sequential[0])
    for modules in (reassemble, model.decoder.convs, model.decoder.fusions):
        modules[-1] = Float32(modules[-1])
    model.head[-2] = Float32(model.head[-2])
