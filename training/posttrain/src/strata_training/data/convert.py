import json
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.data.exr import read_rgb
from strata_training.data.metadata import CONTRACT
from strata_training.io import write_json


def convert_v004(project: Path, output: Path) -> int:
    samples: list[dict[str, Any]] = json.loads((project / "data/v004/samples.json").read_text())
    for sample in samples:
        shot = output / sample["id"]
        render = project / "renders/v004_dataset" / sample["id"]
        for kind in ("beauty", "deep"):
            target = shot / kind / "0001.exr"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(render / f"{kind}.0001.exr")
        _, height, width = read_rgb(render / "beauty.0001.exr", legacy=True).shape
        position = np.array(sample["camera_position"])
        forward = np.array(sample["look_at"]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        rotation = np.stack((right, np.cross(right, forward), -forward))
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = -rotation @ position
        focal = sample["focal_mm"] / sample["aperture_mm"] * width
        camera = {
            "contract": CONTRACT,
            "convention": "USD/OpenGL, column vectors, -Z forward, Y up",
            "move": "locked",
            "frames": [
                {
                    "frame": 1,
                    "width": width,
                    "height": height,
                    "fx": focal,
                    "fy": focal,
                    "cx": width / 2,
                    "cy": height / 2,
                    "world_to_camera": matrix.tolist(),
                    "distortion": {"model": "none"},
                    "shutter_open": -0.25 if sample["category"] == "mb" else 0,
                    "shutter_close": 0.25 if sample["category"] == "mb" else 0,
                    "fstop": sample["fstop"] if sample["category"] == "dof" else 0,
                    "focus_distance_m": sample["distance_m"]
                    + (4 if sample["focus"] == "background" else 0),
                    "focal_length_mm": sample["focal_mm"],
                    "sensor_width_mm": sample["aperture_mm"],
                    "focus_distance_source": "build_dataset_v004.py focusDistance expression",
                }
            ],
        }
        manifest = {
            "contract": CONTRACT,
            "shot_id": sample["id"],
            "seed": sample["seed"],
            "recipe_version": "legacy-v004-adapter-0.1.0",
            "generator_versions": {"source": "unversioned-v004-spike", "adapter": "0.1.0"},
            "spec": {"source": "v004", "pose": sample["pose"]},
            "scene": {"legacy": sample},
            "instance_ids": [],
            "assets": [
                {
                    "id": "v004-render",
                    "source": "local-generated-render",
                    "license": "source asset licenses not audited by this adapter",
                    "origin": str(render),
                }
            ],
            "render": {
                "renderer": "Karma CPU",
                "width": width,
                "height": height,
                "path_traced_samples": sample["spp"],
                "frames": [1, 1],
                "aovs": ["C"],
                "deep_aovs": ["R", "G", "B", "A", "Z", "ZBack"],
            },
            "timings": {},
            "split": sample["split"],
            "category": sample["category"],
            "color_space": "legacy_scene_linear_unverified",
            "legacy": sample,
            "legacy_data": str(project / "data/v004" / sample["id"]),
            "adapter_image_conformance": False,
            "missing_aovs": [
                "albedo",
                "N",
                "roughness",
                "metallic",
                "lighting_splits",
                "instance_id",
            ],
        }
        write_json(shot / "manifest.json", manifest)
        write_json(shot / "camera.json", camera)
    return len(samples)
