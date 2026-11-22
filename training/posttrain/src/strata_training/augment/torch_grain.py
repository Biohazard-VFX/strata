"""Torch implementation of the Nuke-matched Alexa grain, including its noise field."""

import torch
import torch.nn.functional as F

from strata_training.augment.grain import (
    FILM_RESPONSE,
    ISO_STOPS,
    NOISE_CHANNEL,
    NOISE_GAMMA,
    NOISE_ZOFFSET,
    PROFILES,
    SHARPEN_KERNEL,
    GrainParams,
    exposure_level,
)
from strata_training.augment.noise import GRADIENTS, PERIOD, PERMUTATION, ROTATION, octave_count
from strata_training.torch_api import api


def _gradient_noise(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(PERMUTATION, device=x.device, dtype=torch.long)
    gradients = torch.as_tensor(GRADIENTS, device=x.device)
    fx, fy, fz = x.floor(), y.floor(), z.floor()
    ix, iy, iz = (v.long() & (PERIOD - 1) for v in (fx, fy, fz))
    x, y, z = x - fx, y - fy, z - fz
    wx = (1 - x * x * (3 - 2 * x), x * x * (3 - 2 * x))
    wy = (1 - y * y * (3 - 2 * y), y * y * (3 - 2 * y))
    w = z * z * (3 - 2 * z)
    xs, ys = (x, x - 1), (y, y - 1)
    a, b = table[ix] + iy, table[ix + 1] + iy
    total = torch.zeros_like(x)
    for base, dx, dy in ((a, 0, 0), (b, 1, 0), (a + 1, 0, 1), (b + 1, 1, 1)):
        corner = table[base] + iz
        near, far = gradients[corner & 15], gradients[(corner + 1) & 15]
        planar = near[..., 0] * xs[dx] + near[..., 1] * ys[dy] + near[..., 2] * z
        back = far[..., 0] * xs[dx] + far[..., 1] * ys[dy] + far[..., 2] * (z - 1)
        total = total + wx[dx] * wy[dy] * (planar + w * (back - planar))
    return total


_compiled_noise = api.compile(_gradient_noise, fullgraph=True, dynamic=True)


def gradient_noise(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if x.device.type == "cuda":
        return _compiled_noise(x, y, z)
    return _gradient_noise(x, y, z)


def noise_node(
    shape: tuple[int, int], size: float, zoffset: float, gamma: float, device: torch.device
) -> torch.Tensor:
    height, width = shape
    cols = torch.arange(width, device=device, dtype=torch.float64) / size
    rows = (height - 1 - torch.arange(height, device=device, dtype=torch.float64)) / size
    x = cols[None].expand(shape)
    y = rows[:, None].expand(shape)
    plane = torch.stack((x, y, torch.full_like(x, zoffset)))
    rotation = torch.tensor(ROTATION, device=device)
    point = torch.einsum("ij,jhw->ihw", rotation, plane).float()
    total = torch.zeros_like(point[0])
    for octave in range(octave_count(size)):
        scaled = point * 2.0**octave
        total = total + gradient_noise(scaled[0], scaled[1], scaled[2])
    value = 0.5 + 0.5 * total
    return value if gamma == 1 else torch.where(value > 0, value.abs() ** (1 / gamma), value)


def blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Borrow BCHW values, return separable nearest-edge convolution on their device."""
    channels = image.shape[1]
    radius = kernel.numel() // 2
    vertical = kernel[None, None, :, None].expand(channels, 1, -1, 1)
    horizontal = kernel[None, None, None, :].expand(channels, 1, 1, -1)
    result = F.conv2d(
        F.pad(image, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels
    )
    return F.conv2d(
        F.pad(result, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels
    )


def response(image: torch.Tensor, iso: int) -> torch.Tensor:
    stops = ISO_STOPS[iso]
    channels: list[torch.Tensor] = []
    for channel, curve in enumerate(FILM_RESPONSE):
        value = image[channel].double() * 2.0**-stops
        xs = torch.tensor(curve.table_x, device=image.device)
        ys = torch.tensor(curve.table_y, device=image.device)
        indices = torch.searchsorted(xs, value.contiguous()).clamp(1, xs.numel() - 1)
        low, high = indices - 1, indices
        inside = ys[low] + (value - xs[low]) / (xs[high] - xs[low]).clamp_min(1e-300) * (
            ys[high] - ys[low]
        )
        below = float(curve.ys[0]) + (value - float(curve.xs[0])) * float(curve.slopes[0])
        above = float(curve.ys[-1]) + (value - float(curve.xs[-1])) * float(curve.slopes[-1])
        result = torch.where(
            value < float(curve.xs[0]),
            below,
            torch.where(value > float(curve.xs[-1]), above, inside),
        )
        channels.append(result * 2.0**stops)
    return torch.stack(channels).float()


def apply_grain(image: torch.Tensor, params: GrainParams) -> torch.Tensor:
    """Borrow CHW linear RGB; return deterministic device-owned Alexa grain values."""
    profile = PROFILES[params.profile]
    level = exposure_level(params.exposure_offset)
    noise = torch.zeros_like(image)
    for size, amplitude, gamma, zoffset, channel in zip(
        profile.sizes, profile.amplitudes, NOISE_GAMMA, NOISE_ZOFFSET, NOISE_CHANNEL, strict=True
    ):
        field = (
            noise_node(
                (image.shape[-2], image.shape[-1]), size, params.seed + zoffset, gamma, image.device
            )
            - 0.5
        )
        if channel is None:
            noise = noise + field * (amplitude * level)
        else:
            noise[channel] += field * (amplitude * level)
    gain = torch.tensor(profile.channel_gain, device=image.device)[:, None, None]
    noise = noise * gain
    kernel = torch.tensor(SHARPEN_KERNEL, device=image.device, dtype=noise.dtype)
    blurred = blur(noise[None], kernel)[0]
    amount = torch.tensor(profile.sharpen, device=image.device)[:, None, None]
    noise = (noise + amount * (noise - blurred)) * params.amount
    return image + noise * response(image, params.iso)
