"""Contact sheet of the first frame of every shot, for a quick visual review."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from strata_datagen.exr import read_flat

THUMB = (640, 360)
COLUMNS = 4


def _display(path: Path) -> Image.Image:
    beauty = read_flat(path)
    rgb = np.stack([beauty[c] for c in "RGB"], axis=-1).astype(np.float64)
    rgb = np.nan_to_num(rgb) * (0.18 / max(float(np.median(rgb.mean(axis=-1))), 1e-4))
    mapped = rgb / (1.0 + rgb)
    encoded = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1 / 2.4) - 0.055)
    image = Image.fromarray(np.round(np.clip(encoded, 0, 1) * 255).astype(np.uint8))
    return image.resize(THUMB, Image.Resampling.LANCZOS)


def build(shots: list[tuple[Path, str]], output: Path) -> Path:
    rows = (len(shots) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (THUMB[0] * COLUMNS, THUMB[1] * rows), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (shot, label) in enumerate(shots):
        x, y = THUMB[0] * (index % COLUMNS), THUMB[1] * (index // COLUMNS)
        sheet.paste(_display(shot / "beauty" / "0001.exr"), (x, y))
        draw.rectangle((x, y, x + THUMB[0], y + 22), fill=(0, 0, 0))
        draw.text((x + 6, y + 5), label, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return output
