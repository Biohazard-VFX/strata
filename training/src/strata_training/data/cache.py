import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from strata_training.data.exr import read_deep, read_rgb
from strata_training.data.targets import composite
from strata_training.types import Array, Layers, fp16_safe


def prepare(dataset: Path, output: Path, layers: int = 4) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(dataset.glob("*/manifest.json")):
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        shot = manifest_path.parent
        for frame in manifest["frames"]:
            identifier = f"{shot.name}_{int(frame):04d}"
            cache = output / f"{identifier}.npz"
            legacy = Path(manifest["legacy_data"]) if "legacy_data" in manifest else None
            if cache.exists() and np.load(cache)["alpha"].shape[0] != layers:
                raise ValueError("Target cache K changed; use a new cache directory")
            if not cache.exists():
                target = read_deep(shot / "deep" / f"{int(frame):04d}.exr", layers)
                image = read_rgb(shot / "beauty" / f"{int(frame):04d}.exr")
                if legacy is not None:
                    srgb = np.asarray(Image.open(legacy / "beauty.png").convert("RGB")).transpose(
                        2, 0, 1
                    )
                    old = np.load(legacy / "gt.npz")
                    mixed, fur = old["mixed"], old["fur"]
                    reference = {f"reference_{k}": old[k] for k in ("a_fg", "inv_fg", "inv_bg")}
                else:
                    srgb = display_rgb(image, manifest.get("color_space", "ACEScg"))
                    total = 1 - np.prod(1 - target.alpha, axis=0)
                    mixed = (
                        (target.valid.sum(axis=0) > 1)
                        & (target.alpha[0] > 0.03)
                        & (target.alpha[0] < 0.97)
                        & (total > 0.99)
                    )
                    fur = np.zeros_like(mixed)
                    reference = {}
                np.savez_compressed(
                    cache,
                    inverse_depth=target.inverse_depth,
                    alpha=target.alpha,
                    rgb=fp16_safe(target.rgb),
                    beauty=fp16_safe(image),
                    srgb=srgb,
                    mixed=mixed,
                    fur=fur,
                    **reference,
                )
                error = float(np.abs(composite(target.alpha, target.rgb) - image).mean())
                print(
                    json.dumps(
                        {"id": identifier, "deep_beauty_mae": error, "mixed": int(mixed.sum())}
                    ),
                    flush=True,
                )
            entries.append(
                {
                    "id": identifier,
                    "shot": shot.name,
                    "frame": int(frame),
                    "split": manifest["split"],
                    "category": manifest["category"],
                    "cache": str(cache.resolve()),
                    "manifest": str(manifest_path.resolve()),
                    "legacy_data": str(legacy) if legacy else None,
                }
            )
    if not entries:
        raise ValueError(f"No shot manifests found in {dataset}")
    (output / "index.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


def display_rgb(image: Array, color_space: str) -> Array:
    if color_space == "ACEScg":
        matrix = np.array(
            [
                [1.70505, -0.62179, -0.08326],
                [-0.13026, 1.1408, -0.01055],
                [-0.024, -0.12897, 1.15297],
            ]
        )
        image = np.einsum("ij,jhw->ihw", matrix, image)
    image = np.clip(image * 0.71619818, 0, 1)
    display = np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)
    return (display * 255 + 0.5).astype(np.uint8)


def load_layers(path: Path) -> tuple[Layers, dict[str, Array]]:
    with np.load(path) as file:
        arrays = {k: file[k] for k in file.files}
    return Layers(
        arrays["inverse_depth"], arrays["alpha"], arrays["rgb"].astype(np.float32)
    ), arrays
