import math

import torch

DISPLAY_MATRIX = (
    (1.70505, -0.62179, -0.08326),
    (-0.13026, 1.1408, -0.01055),
    (-0.024, -0.12897, 1.15297),
)


def display(images: torch.Tensor, color_space: str, exposure: float = 0) -> torch.Tensor:
    if not math.isfinite(exposure):
        raise ValueError("Exposure must be finite stops, fixed for the shot")
    if color_space == "ACEScg":
        matrix = torch.tensor(DISPLAY_MATRIX, device=images.device, dtype=torch.float64)
        images = torch.einsum("ij,...jhw->...ihw", matrix, images.double())
    gain = torch.tensor(0.72 * 2.0**exposure, device=images.device, dtype=torch.float64)
    linear = images.double() * gain
    linear = torch.nan_to_num(linear, nan=0, posinf=float("inf"), neginf=0).clamp_min(0).double()
    delta = (linear - 0.6).clamp_min(0)
    shoulder = 0.6 + 0.4 * delta / (0.4 + delta)
    shoulder = torch.nan_to_num(shoulder, nan=1)
    value = torch.where(linear <= 0.6, linear, shoulder)
    return torch.where(
        value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055
    ).float()


def normalize(display_rgb: torch.Tensor) -> torch.Tensor:
    if display_rgb.dtype != torch.float32:
        raise ValueError("Encoder input must be float32; rebuild target caches")
    mean = torch.tensor([0.485, 0.456, 0.406], device=display_rgb.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=display_rgb.device)[:, None, None]
    return (display_rgb - mean) / std
