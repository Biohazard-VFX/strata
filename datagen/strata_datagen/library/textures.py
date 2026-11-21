"""Bring Megascans texture maps to a bounded size as 8-bit files with a known encoding."""

from pathlib import Path

import numpy as np
import OpenEXR
from numpy.typing import NDArray
from PIL import Image

from strata_datagen import files
from strata_datagen.library.megascans import TextureMap

MAX_SIZE = 2048
Pixels = NDArray[np.float32]
OUTPUT = {
    "Albedo": ("albedo.jpg", "RGB"),
    "Normal": ("normal.png", "RGB"),
    "Roughness": ("roughness.png", "L"),
    "Opacity": ("opacity.png", "L"),
}


def _srgb(linear: Pixels) -> Pixels:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def _read_exr(path: Path, channel: str) -> Pixels:
    channels = OpenEXR.File(str(path), separate_channels=True).parts[0].channels
    names = [c for c in "RGB" if c in channels] or sorted(channels)[:1]
    stack = np.stack([np.asarray(channels[c].pixels, np.float32) for c in names], axis=-1)
    return _srgb(stack) if channel == "Albedo" else np.clip(stack, 0.0, 1.0)


def _to_image(pixels: Pixels, mode: str) -> Image.Image:
    data = pixels[..., :1] if mode == "L" else np.repeat(pixels, 3 // pixels.shape[-1], axis=-1)[..., :3]
    image = Image.fromarray(np.round(np.clip(data, 0.0, 1.0) * 255).astype(np.uint8).squeeze())
    return image.convert(mode)


def convert(source: TextureMap, folder: Path) -> Path:
    name, mode = OUTPUT[source.channel]
    if source.path.suffix.lower() == ".exr":
        image = _to_image(_read_exr(source.path, source.channel), mode)
    else:
        with Image.open(source.path) as opened:
            image = opened.convert(mode)
    if max(image.size) > MAX_SIZE:
        image = image.resize(
            (MAX_SIZE * image.width // max(image.size), MAX_SIZE * image.height // max(image.size)),
            Image.Resampling.LANCZOS,
        )
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    with files.replacing(target) as temporary:
        image.save(temporary, quality=93) if target.suffix == ".jpg" else image.save(temporary)
    return target
