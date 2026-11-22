from typing import cast

import torch
from torch import nn
from torch.nn import functional as functional


class DeepHead(nn.Module):
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def __init__(self, channels: int, inverse_head: nn.Module, layers: int = 4) -> None:
        super().__init__()
        self.layers = layers
        self.inverse_head = inverse_head.requires_grad_(False)
        self.stem = nn.Conv2d(channels + 4, 48, 1)
        self.blocks = nn.ModuleList(
            nn.Conv2d(48, 48, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 16, 1)
        )
        self.output = nn.Conv2d(48, layers * 5, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(cast(torch.Tensor, self.output.bias))
        self.register_buffer("input_mean", torch.zeros(channels + 4))
        self.register_buffer("input_std", torch.ones(channels + 4))

    def forward(
        self,
        features: torch.Tensor,
        image: torch.Tensor,
        base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = base.clamp_min(1e-3)
        inputs = torch.cat((features, image, base), dim=1)
        inputs = (inputs - self.input_mean[None, :, None, None]) / self.input_std[
            None, :, None, None
        ]
        x = functional.silu(self.stem(inputs))
        for block in self.blocks:
            x = x + 0.2 * functional.silu(block(x))
        raw = self.output(x).float().reshape(x.shape[0], self.layers, 5, *x.shape[-2:])
        first = base[:, 0] * torch.exp(raw[:, 0, 0].clamp(-8, 8))
        deltas = functional.softplus(raw[:, 1:, 0])
        inverse = torch.cat(
            (first[:, None], first[:, None] * torch.exp(-deltas.cumsum(dim=1))), dim=1
        )
        alpha = raw[:, :, 1].sigmoid()
        color = image[:, None] + raw[:, :, 2:]
        return {
            "inverse_depth": inverse,
            "alpha": alpha,
            "alpha_logits": raw[:, :, 1],
            "rgb": alpha[:, :, None] * color,
        }
