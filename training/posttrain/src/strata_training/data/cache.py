import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from strata_dataset.samples import recorded_identity

from strata_training.data.compact import read_layers
from strata_training.data.exr import read_rgb, read_roles
from strata_training.data.metadata import camera_frames, category
from strata_training.data.policy import is_test_only
from strata_training.io import save_arrays, write_json
from strata_training.types import Array, Layers, fp16_safe

CACHE_VERSION = "targets-content-v1"


def content_identity(
    shot: Path, manifest: dict[str, Any], camera: dict[str, Any], frame: int, layers: int
) -> str:
    hashes: dict[str, str] = {}
    for name in ("layers", "beauty"):
        with (shot / name / f"{frame:04d}.exr").open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    record = {
        "version": CACHE_VERSION,
        "layers": layers,
        "manifest": manifest,
        "camera": camera,
        "samples": recorded_identity(shot / "samples" / f"{shot.name}_{frame:04d}"),
        "files": hashes,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


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
            identity = content_identity(shot, manifest, camera, int(frame), layers)
            cache = output / f"{identifier}.{identity}.npz"
            reusable = False
            if cache.exists():
                with np.load(cache) as existing:
                    reusable = (
                        "content_identity" in existing
                        and str(existing["content_identity"]) == identity
                    )
            if not reusable:
                packed = shot / "layers" / f"{int(frame):04d}.exr"
                target = read_layers(packed, layers)
                image = read_rgb(shot / "beauty" / f"{int(frame):04d}.exr")
                expected = (camera["height"], camera["width"])
                if image.shape[-2:] != expected or target.alpha.shape[-2:] != expected:
                    raise ValueError(
                        f"Native beauty/deep/camera dimensions disagree in {identifier}"
                    )
                srgb = display_rgb(image, manifest.get("color_space", "ACEScg"))
                mixed = (target.alpha[0] > 0.03) & (target.alpha[0] < 0.97)
                reference = read_roles(packed)
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
                    content_identity=np.array(identity),
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
                    "recomposition_eligible": recorded_identity(shot / "samples" / identifier)
                    is not None,
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
