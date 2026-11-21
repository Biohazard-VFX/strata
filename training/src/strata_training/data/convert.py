import json
from pathlib import Path
from typing import Any

import numpy as np


def convert_v004(project: Path, output: Path) -> int:
    samples: list[dict[str, Any]] = json.loads((project / "data/v004/samples.json").read_text())
    for sample in samples:
        shot = output / sample["id"]
        for kind in ("beauty", "deep"):
            target = shot / kind / "0001.exr"
            target.parent.mkdir(parents=True, exist_ok=True)
            source = project / "renders/v004_dataset" / sample["id"] / f"{kind}.0001.exr"
            if not target.exists():
                target.symlink_to(source)
        position = np.array(sample["camera_position"])
        forward = np.array(sample["look_at"]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = np.stack((right, up, -forward))
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = -rotation @ position
        focal = sample["focal_mm"] / sample["aperture_mm"] * sample["width"]
        camera = {
            "frames": {
                "0001": {
                    "intrinsics": [[focal, 0, 640], [0, focal, 360], [0, 0, 1]],
                    "world_to_camera": matrix.tolist(),
                    "shutter": [-0.25, 0.25] if sample["category"] == "mb" else [0, 0],
                    "f_stop": sample["fstop"] if sample["category"] == "dof" else None,
                    "focus_distance_m": sample["distance_m"]
                    + (4 if sample["focus"] == "background" else 0),
                    "camera_convention": "right-handed, -Z forward, Y up",
                    "focus_distance_source": "legacy distance_m approximation; source HIP is authoritative",
                }
            }
        }
        manifest = {
            "schema_version": "strata-dataset-v0",
            "shot_id": sample["id"],
            "frames": [1],
            "split": sample["split"],
            "category": sample["category"],
            "width": 1280,
            "height": 720,
            "color_space": "legacy_scene_linear_unverified",
            "source": "v004",
            "legacy": sample,
            "legacy_data": str(project / "data/v004" / sample["id"]),
            "missing_aovs": [
                "albedo",
                "N",
                "roughness",
                "metallic",
                "lighting_splits",
                "instance_id",
            ],
        }
        (shot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (shot / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    return len(samples)
