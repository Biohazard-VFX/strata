from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from strata_training.baselines.deep import two_layer
from strata_training.types import Array, Layers


class FrozenProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(132, 48, 1)
        self.blocks = nn.ModuleList(
            nn.Conv2d(48, 48, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 16, 1)
        )
        self.output = nn.Conv2d(48, 3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = functional.silu(self.stem(inputs))
        for block in self.blocks:
            x = x + 0.2 * functional.silu(block(x))
        output = self.output(x)
        return torch.cat((output[:, :1].sigmoid(), output[:, 1:]), dim=1)


def predict_probe(
    checkpoint: Path,
    normalization: Path,
    srgb: Array,
    inverse: Array,
    features: Array,
    beauty: Array,
) -> Layers:
    model = FrozenProbe().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    norm = np.load(normalization)
    output = np.empty((3, *inverse.shape), dtype=np.float32)
    height, width = inverse.shape
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            y0, x0 = max(y - 32, 0), max(x - 32, 0)
            y1, x1 = min(y + 288, height), min(x + 288, width)
            inputs = np.concatenate(
                (
                    srgb[:, y0:y1, x0:x1].astype(np.float32) / 255,
                    inverse[None, y0:y1, x0:x1],
                    features[:, y0:y1, x0:x1],
                )
            ).astype(np.float32)
            inputs = (inputs - norm["mean"][:, None, None]) / norm["std"][:, None, None]
            with torch.no_grad():
                prediction: torch.Tensor = model(torch.as_tensor(inputs)[None])
                values = prediction[0].numpy()
            h, w = min(256, height - y), min(256, width - x)
            output[:, y : y + h, x : x + w] = values[:, y - y0 : y - y0 + h, x - x0 : x - x0 + w]
    front, back = np.exp(output[1:]) * inverse
    return two_layer(output[0], front, back, beauty)
