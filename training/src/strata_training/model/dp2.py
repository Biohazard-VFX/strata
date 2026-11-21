import copy
from importlib import import_module
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from strata_training.model.head import DeepHead


class DeepDepthPro(nn.Module):
    def __init__(
        self, preset: str = "depthpro2-s", layers: int = 4, decoder_trainable: bool = True
    ) -> None:
        super().__init__()
        upstream: Any = import_module("depth_pro_2.model")
        config = upstream.PRESETS[preset]
        self.base: Any = upstream.DepthPro2(
            backbone_preset=config.backbone_preset,
            reassembled_blocks=config.reassembled_blocks,
            cross_frame_blocks=config.cross_frame_blocks,
            dims_reassemble=config.dims_reassemble,
            dim_decoder=config.dim_decoder,
            fov=False,
            precision=torch.bfloat16,
        )
        self.deep_head = DeepHead(
            config.dim_decoder // 2, copy.deepcopy(self.base.head[2:]), layers
        )
        state: dict[str, torch.Tensor] = upstream.load_checkpoint(config.checkpoint_uri)
        result: Any = self.load_state_dict({f"base.{k}": v for k, v in state.items()}, strict=False)
        if result.unexpected_keys or any(
            not str(k).startswith("deep_head.") for k in result.missing_keys
        ):
            raise ValueError(f"Unexpected DP2 checkpoint incompatibility: {result}")
        self.deep_head.inverse_head.load_state_dict(self.base.head[2:].state_dict())
        self.base.requires_grad_(False)
        self.base.decoder.requires_grad_(decoder_trainable)
        self.base.head[:2].requires_grad_(decoder_trainable)
        self.decoder_trainable = decoder_trainable
        self.preset = preset

    def train(self, mode: bool = True) -> "DeepDepthPro":
        super().train(mode)
        self.base.encoder.eval()
        return self

    def pyramid(self, normalized: torch.Tensor) -> list[torch.Tensor]:
        with (
            torch.no_grad(),
            torch.autocast(normalized.device.type, dtype=torch.bfloat16, cache_enabled=False),
        ):
            tokens, grid = self.base.encoder.encode(normalized, None)
            return list(self.base.encoder.reassemble_tokens(tokens, grid))

    def decode(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        decoded: torch.Tensor = self.base.decoder(pyramid)
        with torch.autocast(decoded.device.type, enabled=False):
            return self.base.head[:2](decoded.float())

    def from_pyramid(
        self, pyramid: list[torch.Tensor], image: torch.Tensor, metric_scale: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features = self.decode(pyramid)
        with torch.autocast(features.device.type, enabled=False):
            base = self.deep_head.inverse_head(features.float()).clamp_min(1e-5) * metric_scale
        return self.deep_head(features, image, base)

    def forward(
        self, normalized: torch.Tensor, image: torch.Tensor, metric_scale: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        height, width = image.shape[-2:]
        padding = (0, -width % 32, 0, -height % 32)
        normalized = functional.pad(normalized, padding, mode="replicate")
        image = functional.pad(image, padding, mode="replicate")
        result = self.from_pyramid(self.pyramid(normalized), image, metric_scale)
        return {k: v[..., :height, :width] for k, v in result.items()}
