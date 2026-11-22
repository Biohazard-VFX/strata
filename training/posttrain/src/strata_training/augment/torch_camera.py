"""Device camera math; only actual final-image codec encoding remains on CPU."""

import math

import torch

from strata_training.augment.camera import (
    AP1_LUMINANCE,
    AP1_TO_AWG3,
    AP1_TO_REC709,
    CameraParams,
    jpeg,
    video,
)
from strata_training.augment.torch_grain import blur


def _matrix(image: torch.Tensor, values: object) -> torch.Tensor:
    return torch.einsum(
        "ij,jhw->ihw", torch.as_tensor(values, device=image.device, dtype=image.dtype), image
    )


def encode(image: torch.Tensor, params: CameraParams) -> torch.Tensor:
    luminance = (
        torch.tensor(AP1_LUMINANCE, device=image.device, dtype=image.dtype)[:, None, None] * image
    ).sum(0)
    graded = luminance + params.saturation * (image - luminance)
    positive = graded.clamp_min(0)
    image = torch.where(graded > 0, 0.18 * (positive / 0.18) ** params.contrast, graded)
    if params.curve == "logc3":
        image = _matrix(image, AP1_TO_AWG3)
        code = torch.where(
            image > 0.010591,
            0.247190 * (5.555556 * image + 0.052272).clamp_min(1e-10).log10() + 0.385537,
            5.367655 * image + 0.092809,
        )
    else:
        image = _matrix(image, AP1_TO_REC709).clamp(0, params.white)
        image = image * (1 + image / params.white**2) / (1 + image)
        code = torch.where(image < 0.018, 4.5 * image, 1.099 * image**0.45 - 0.099)
    if params.sharpen_amount > 0:
        radius = int(4 * params.sharpen_sigma + 0.5)
        x = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float64)
        kernel = (-0.5 * (x / params.sharpen_sigma) ** 2).exp()
        kernel = (kernel / kernel.sum()).to(code.dtype)
        code = code + params.sharpen_amount * (code - blur(code[None], kernel)[0])
    return code


def decode(code: torch.Tensor, params: CameraParams) -> torch.Tensor:
    if params.curve == "logc3":
        image = torch.where(
            code > 5.367655 * 0.010591 + 0.092809,
            (10 ** ((code - 0.385537) / 0.247190) - 0.052272) / 5.555556,
            (code - 0.092809) / 5.367655,
        )
        matrix = torch.tensor(AP1_TO_AWG3, device=code.device, dtype=code.dtype).inverse()
    else:
        code = code.clamp(0, 1)
        display = torch.where(code < 0.081, code / 4.5, ((code + 0.099) / 1.099) ** (1 / 0.45))
        k = 1 / params.white**2
        rest = 1 - display
        image = ((rest * rest + 4 * k * display).sqrt() - rest) / (2 * k)
        matrix = torch.tensor(AP1_TO_REC709, device=code.device, dtype=code.dtype).inverse()
    return _matrix(image, matrix)


def codec(code: torch.Tensor, params: CameraParams) -> torch.Tensor:
    """Borrow encoded CHW; CPU codecs see only the final cropped input, never labels."""
    if params.codec == "none":
        levels = math.pow(2, params.bits) - 1
        return (code.clamp(0, 1) * levels).round() / levels
    cpu = code.detach().cpu().numpy()
    result = (
        jpeg(cpu, params.quality)
        if params.codec == "jpeg"
        else video(cpu, params.codec, params.quality, params.bits)
    )
    return torch.tensor(result, device=code.device)


def degrade(image: torch.Tensor, params: CameraParams) -> torch.Tensor:
    return decode(codec(encode(image, params), params), params).float()
