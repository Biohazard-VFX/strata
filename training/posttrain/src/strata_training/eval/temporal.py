from pathlib import Path
from typing import Any

import numpy as np

from strata_training.data.metadata import camera_frames, intrinsics
from strata_training.types import Array


def static_correspondences(
    depth0: Array,
    depth1: Array,
    intrinsics0: Array,
    intrinsics1: Array,
    world_to_camera0: Array,
    world_to_camera1: Array,
    stride: int = 4,
) -> tuple[Array, Array, Array, Array]:
    height, width = depth0.shape
    y, x = np.mgrid[:height:stride, :width:stride]
    z = depth0[y, x]
    camera = np.stack(
        (
            (x + 0.5 - intrinsics0[0, 2]) / intrinsics0[0, 0] * z,
            -(y + 0.5 - intrinsics0[1, 2]) / intrinsics0[1, 1] * z,
            -z,
            np.ones_like(z),
        )
    )
    transformed = (
        world_to_camera1 @ np.linalg.inv(world_to_camera0) @ camera.reshape(4, -1)
    ).reshape(camera.shape)
    projected_z = -transformed[2]
    denominator = np.maximum(projected_z, 1e-8)
    px = np.rint(intrinsics1[0, 0] * transformed[0] / denominator + intrinsics1[0, 2] - 0.5).astype(
        int
    )
    py = np.rint(
        -intrinsics1[1, 1] * transformed[1] / denominator + intrinsics1[1, 2] - 0.5
    ).astype(int)
    valid = (z > 0) & (projected_z > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py = np.clip(px, 0, width - 1), np.clip(py, 0, height - 1)
    valid &= np.abs(depth1[py, px] / denominator - 1) <= 0.02
    return y[valid], x[valid], py[valid], px[valid]


def sequence_jitter(entries: list[dict[str, Any]], predictions: Path) -> dict[str, Any]:
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_shot.setdefault(entry["shot"], []).append(entry)
    reports: list[dict[str, Any]] = []
    for shot, frames in by_shot.items():
        frames.sort(key=lambda e: e["frame"])
        if len(frames) < 2:
            continue
        camera = camera_frames(Path(frames[0]["manifest"]).parent)
        residuals: list[Array] = []
        depth: list[Array] = []
        alpha_errors: list[Array] = []
        rgb_errors: list[Array] = []
        for entry in frames:
            gt = np.load(entry["cache"])
            pred = np.load(predictions / f"{entry['id']}.npz")
            true_z = np.where(
                gt["inverse_depth"][0] > 0, 1 / np.maximum(gt["inverse_depth"][0], 1e-8), 0
            )
            predicted_z = 1 / np.maximum(pred["inverse_depth"][0], 1e-8)
            residuals.append(np.log(predicted_z) - np.log(np.maximum(true_z, 1e-8)))
            depth.append(true_z)
            alpha_errors.append(pred["alpha"][0] - gt["alpha"][0])
            rgb_errors.append(pred["rgb"][0].astype(np.float32) - gt["rgb"][0].astype(np.float32))
        depth_deltas: list[Array] = []
        alpha_deltas: list[Array] = []
        color_deltas: list[Array] = []
        for i in range(len(frames) - 1):
            if frames[i + 1]["frame"] != frames[i]["frame"] + 1:
                continue
            first, second = (camera[e["frame"]] for e in frames[i : i + 2])
            y0, x0, y1, x1 = static_correspondences(
                depth[i],
                depth[i + 1],
                intrinsics(first),
                intrinsics(second),
                np.asarray(first["world_to_camera"]),
                np.asarray(second["world_to_camera"]),
            )
            depth_deltas.append(np.abs(residuals[i + 1][y1, x1] - residuals[i][y0, x0]))
            alpha_deltas.append(np.abs(alpha_errors[i + 1][y1, x1] - alpha_errors[i][y0, x0]))
            color_deltas.append(
                np.abs(rgb_errors[i + 1][:, y1, x1] - rgb_errors[i][:, y0, x0]).mean(axis=0)
            )
        if depth_deltas and sum(a.size for a in depth_deltas):
            reports.append(
                {
                    "shot": shot,
                    "tracks": sum(a.size for a in depth_deltas),
                    "log_depth_residual_jitter": float(np.concatenate(depth_deltas).mean()),
                    "alpha_residual_jitter": float(np.concatenate(alpha_deltas).mean()),
                    "rgb_residual_jitter": float(np.concatenate(color_deltas).mean()),
                }
            )
    return {
        "sequences": reports,
        "available": bool(reports),
        "protocol": (
            "Nearest native-pixel static reprojection, stride4, GT depth agreement <=2%; "
            "front-layer residual differences, no per-frame alignment. "
            "Not dynamic per-sample identity tracks."
        ),
    }
