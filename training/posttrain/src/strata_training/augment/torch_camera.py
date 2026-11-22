"""Device camera math; only actual final-image codec encoding remains on CPU."""

from concurrent.futures import ThreadPoolExecutor

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
from strata_training.types import Array


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


def _codec_array(code: Array, params: CameraParams) -> Array:
    if params.codec == "jpeg":
        return jpeg(code, params.quality)
    return video(code, params.codec, params.quality, params.bits)


def degrade_batch(images: torch.Tensor, params: list[CameraParams]) -> torch.Tensor:
    """Borrow BCHW inputs; CPU workers encode only final images, never supervision.

    Each batch owns at most one codec thread per compressed input and joins before return.
    """
    codes = [encode(image, p) for image, p in zip(images, params, strict=True)]
    compressed = [i for i, p in enumerate(params) if p.codec != "none"]
    if compressed:
        arrays = {i: codes[i].detach().cpu().numpy() for i in compressed}
        with ThreadPoolExecutor(max_workers=len(compressed)) as workers:
            futures = {i: workers.submit(_codec_array, arrays[i], params[i]) for i in compressed}
            for i in compressed:
                codes[i] = torch.tensor(futures[i].result(), device=images.device)
    for i, p in enumerate(params):
        if p.codec == "none":
            levels = (1 << p.bits) - 1
            codes[i] = (codes[i].clamp(0, 1) * levels).round() / levels
    return torch.stack([decode(code, p).float() for code, p in zip(codes, params, strict=True)])


def degrade(image: torch.Tensor, params: CameraParams) -> torch.Tensor:
    return degrade_batch(image[None], [params])[0]
