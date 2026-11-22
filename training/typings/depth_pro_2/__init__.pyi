from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

class Rope(nn.Module):
    periods: Tensor
    rotate_half: bool

class Encoder(nn.Module):
    rope: Rope

class MultiViewEncoder(Encoder):
    blocks: nn.Sequential
    reassemble: nn.ModuleList

class DPTDecoder(nn.Module):
    convs: nn.ModuleList
    fusions: nn.ModuleList

class FOVHead(nn.Module):
    def forward(self, tokens: Tensor, grid_size: tuple[int, int]) -> Tensor: ...

class FOVEstimator(nn.Module):
    precision: torch.dtype
    trunk: Encoder
    head: FOVHead
    def encode(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]: ...
    def focal_length_px(self, image: Tensor) -> Tensor: ...

class Prediction(TypedDict):
    inverse_depth: Tensor
    depth: Tensor | None
    focallength_px: Tensor | None

class DepthPro2(nn.Module):
    precision: torch.dtype
    encoder: MultiViewEncoder
    decoder: DPTDecoder
    head: nn.Sequential
    fov: FOVEstimator | None
    def __init__(
        self,
        backbone_preset: str,
        reassembled_blocks: Sequence[int],
        cross_frame_blocks: Sequence[int],
        dims_reassemble: Sequence[int],
        dim_decoder: int,
        fov: bool = True,
        precision: torch.dtype = torch.float32,
    ) -> None: ...
    def forward(self, image: Tensor) -> Tensor: ...
    def infer(self, image: Tensor, f_px: float | None = None) -> Prediction: ...

def create_model_and_transforms(
    config: str = "depthpro2-h",
    device: str | None = None,
    precision: torch.dtype | None = None,
) -> tuple[DepthPro2, Callable[[NDArray[np.uint8]], Tensor]]: ...
def load_rgb(path: str) -> tuple[NDArray[np.uint8], float | None]: ...
