import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from strata_training.data.compact import read_layers
from strata_training.data.exr import read_deep, read_rgb, read_roles
from strata_training.data.metadata import CONTRACT, camera_frames, category
from strata_training.data.policy import is_test_only
from strata_training.io import save_arrays, write_json
from strata_training.types import Array, Layers, fp16_safe


def prepare(dataset: Path, output: Path, layers: int = 4) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    manifests = sorted(dataset.glob("*/manifest.json"))
    ranked = sorted(manifests, key=lambda p: hashlib.sha256(p.parent.name.encode()).digest())
    heldout: set[str] = (
        {p.parent.name for p in ranked[: max(1, len(ranked) // 5)]} if len(ranked) > 1 else set()
    )
    for manifest_path in manifests:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        if manifest.get("contract") != CONTRACT:
            raise ValueError(f"Unsupported manifest contract in {manifest_path}")
        shot = manifest_path.parent
        if manifest["shot_id"] != shot.name:
            raise ValueError("Shot directory does not match its literal shot_id")
        protected = is_test_only(shot)
        split = (
            "test"
            if protected
            else manifest.get("split", "test" if shot.name in heldout else "train")
        )
        if split not in {"train", "test"}:
            raise ValueError("Shot split must be train or test")
        for frame, camera in sorted(camera_frames(shot).items()):
            identifier = f"{shot.name}_{int(frame):04d}"
            cache = output / f"{identifier}.npz"
            legacy = Path(manifest["legacy_data"]) if "legacy_data" in manifest else None
            if cache.exists():
                with np.load(cache) as existing:
                    if existing["alpha"].shape[0] != layers:
                        raise ValueError("Target cache K changed; use a new cache directory")
                    if legacy is None and (
                        "role_mask_version" not in existing
                        or str(existing["role_mask_version"]) != "linear-roles-v3"
                    ):
                        raise ValueError("Native role masks changed; use a new cache directory")
            if not cache.exists():
                packed = shot / "layers" / f"{int(frame):04d}.exr"
                deep = shot / "deep" / f"{int(frame):04d}.exr"
                target = (
                    read_layers(packed, layers)
                    if packed.exists()
                    else read_deep(deep, layers, background="background" in manifest)
                )
                image = read_rgb(
                    shot / "beauty" / f"{int(frame):04d}.exr", legacy=legacy is not None
                )
                expected = (camera["height"], camera["width"])
                if image.shape[-2:] != expected or target.alpha.shape[-2:] != expected:
                    raise ValueError(
                        f"Native beauty/deep/camera dimensions disagree in {identifier}"
                    )
                if legacy is not None:
                    srgb = np.asarray(Image.open(legacy / "beauty.png").convert("RGB")).transpose(
                        2, 0, 1
                    )
                    old = np.load(legacy / "gt.npz")
                    mixed, fur = old["mixed"], old["fur"]
                    reference = {f"reference_{k}": old[k] for k in ("a_fg", "inv_fg", "inv_bg")}
                else:
                    srgb = display_rgb(image, manifest.get("color_space", "ACEScg"))
                    mixed = (target.alpha[0] > 0.03) & (target.alpha[0] < 0.97)
                    reference = read_roles(packed)
                    fur = reference.pop("fur")
                if "background" in manifest:
                    reference["denoised"] = fp16_safe(
                        read_rgb(shot / "beauty" / f"{int(frame):04d}.exr", part_name="denoised")
                    )
                save_arrays(
                    cache,
                    inverse_depth=target.inverse_depth,
                    alpha=target.alpha,
                    rgb=fp16_safe(target.rgb),
                    beauty=fp16_safe(image),
                    srgb=srgb,
                    mixed=mixed,
                    **({"fur": fur} if fur is not None else {}),
                    role_mask_version=np.array("legacy-v004" if legacy else "linear-roles-v3"),
                    **reference,
                )
            entries.append(
                {
                    "id": identifier,
                    "shot": shot.name,
                    "frame": int(frame),
                    "split": split,
                    "test_only": protected,
                    "dataset_root": str(dataset.resolve()),
                    "category": manifest.get("category", category(camera)),
                    "cache": str(cache.resolve()),
                    "manifest": str(manifest_path.resolve()),
                    "legacy_data": str(legacy) if legacy else None,
                    "compact_scene_only": False,
                }
            )
    if not entries:
        raise ValueError(f"No shot manifests found in {dataset}")
    write_json(output / "index.json", entries)
    return entries


DISPLAY_MATRIX = np.array(
    [[1.70505, -0.62179, -0.08326], [-0.13026, 1.1408, -0.01055], [-0.024, -0.12897, 1.15297]]
)
DISPLAY_EXPOSURE = 0.71619818


def display_rgb(image: Array, color_space: str) -> Array:
    if color_space == "ACEScg":
        image = np.einsum("ij,jhw->ihw", DISPLAY_MATRIX, image)
    image = np.clip(image * DISPLAY_EXPOSURE, 0, 1)
    display = np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)
    return (display * 255 + 0.5).astype(np.uint8)


def load_layers(path: Path) -> tuple[Layers, dict[str, Array]]:
    with np.load(path) as file:
        arrays = {k: file[k] for k in file.files}
    return Layers(
        arrays["inverse_depth"], arrays["alpha"], arrays["rgb"].astype(np.float32)
    ), arrays
