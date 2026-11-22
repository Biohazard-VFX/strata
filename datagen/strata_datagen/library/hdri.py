"""Prepare latlong HDRIs for the dome: resample, cut the sun out into a measured distant light.

Dome frame (USD/Karma DomeLight): +Y is the top row; image column u maps to azimuth
phi = 2*pi*u with direction (sin(theta) sin(phi), cos(theta), -sin(theta) cos(phi)), so the image
centre faces +Z (measured with Karma XPU in Houdini 22.0).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenEXR
from numpy.typing import NDArray

from strata_datagen import files

SOURCE = "Poly Haven (HDRI Haven)"
LICENSE = "CC0"
LUMA = np.array([0.2126, 0.7152, 0.0722])
SUN_CAP_DEG = 2.5
Image = NDArray[np.float32]


@dataclass(frozen=True)
class SunCut:
    direction: tuple[float, float, float]
    irradiance: float
    color: tuple[float, float, float]
    sky_irradiance: float


def read_rgb(path: Path) -> Image:
    channels = OpenEXR.File(str(path), separate_channels=True).parts[0].channels
    return np.stack([np.asarray(channels[c].pixels, np.float32) for c in "RGB"], axis=-1)


def downsample(image: Image, factor: int) -> Image:
    h, w, c = image.shape
    blocks = image[: h - h % factor, : w - w % factor].reshape(h // factor, factor, w // factor, factor, c)
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def directions(height: int, width: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit direction per pixel (h, w, 3) and per-pixel solid angle (h, 1)."""
    theta = np.pi * (np.arange(height) + 0.5) / height
    phi = 2.0 * np.pi * (np.arange(width) + 0.5) / width
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    dirs = np.stack(np.broadcast_arrays(st * np.sin(phi)[None], ct, -st * np.cos(phi)[None]), axis=-1)
    solid = (2.0 * np.pi / width) * (np.pi / height) * np.sin(theta)[:, None]
    return dirs, solid


def cut_sun(image: Image) -> tuple[Image, SunCut]:
    """Clamp the brightest cap to its surrounding sky level and measure what was removed."""
    h, w, _ = image.shape
    dirs, solid = directions(h, w)
    luma = image @ LUMA
    peak = np.unravel_index(int(np.argmax(luma)), luma.shape)
    sun_dir = dirs[peak]
    angle = np.degrees(np.arccos(np.clip(dirs @ sun_dir, -1.0, 1.0)))
    ring = (angle > SUN_CAP_DEG) & (angle < 3 * SUN_CAP_DEG)
    ceiling = float(np.percentile(luma[ring], 99.0))
    cap = angle <= SUN_CAP_DEG
    scale = np.where(cap & (luma > ceiling), ceiling / np.maximum(luma, 1e-12), 1.0)
    clamped = (image * scale[..., None]).astype(np.float32)
    removed = ((image - clamped) * solid[..., None]).reshape(-1, 3).sum(axis=0)
    irradiance = float(removed @ LUMA)
    up = np.maximum(dirs[..., 1], 0.0)
    sky = float(((clamped @ LUMA) * up * solid).sum())
    color = removed / max(irradiance, 1e-9) if irradiance > 0 else np.ones(3)
    return clamped, SunCut(
        direction=(float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])),
        irradiance=irradiance,
        color=(float(color[0]), float(color[1]), float(color[2])),
        sky_irradiance=sky,
    )


def write_rgb(image: Image, path: Path) -> None:
    channels = {c: np.ascontiguousarray(image[..., i]).astype(np.float16) for i, c in enumerate("RGB")}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with files.replacing(path) as temporary:
        OpenEXR.File(header, channels).write(str(temporary))


def prepare(source: Path, destination: Path, factor: int = 2) -> SunCut:
    clamped, cut = cut_sun(downsample(read_rgb(source), factor))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_rgb(clamped, destination)
    return cut
