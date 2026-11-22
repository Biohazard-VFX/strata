"""Weight-sharing frozen encoder for independent, native-sized crop batches."""

import copy
from functools import partial
from typing import Any, cast

import torch


def _independent(block: Any, tokens: torch.Tensor, rope: Any, *, prefix: int) -> torch.Tensor:
    output: torch.Tensor = block(tokens, rope=rope)
    # DP2 discards updated prefix tokens at cross-frame blocks, even for T=1.
    return torch.cat((tokens[:, :prefix], output[:, prefix:]), dim=1)


class FrozenEncoder:
    """Borrow frozen encoder weights; calls are sequential, outputs support head backward.

    The shallow view shares modules, not temporal attention or mutable call state.
    Never use inference tensors here: trainable decoders save these outputs for backward.
    """

    def __init__(self, encoder: Any, *, compiled: bool = False) -> None:
        encoder.eval().requires_grad_(False)
        self.encoder: Any = copy.copy(encoder)
        self.encoder.cross_frame_attention = partial(
            _independent, prefix=int(encoder.num_prefix_tokens)
        )
        if not all(block.attn.fused_attn for block in encoder.blocks):
            raise ValueError("Live encoder requires upstream SDPA attention")
        self.run = torch.compile(self._run, fullgraph=True, dynamic=True) if compiled else self._run

    def _run(self, image: torch.Tensor) -> list[torch.Tensor]:
        tokens, grid = self.encoder.encode(image)
        return cast(list[torch.Tensor], self.encoder.reassemble_tokens(tokens, grid))

    def __call__(self, image: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad(), torch.autocast(image.device.type, dtype=torch.bfloat16):
            return self.run(image)
