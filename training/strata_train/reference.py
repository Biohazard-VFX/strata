"""Generate independent PyTorch oracle arrays without sharing a GPU with ORT."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np
import OpenEXR
import torch
from depth_pro_2 import DepthPro2, create_model_and_transforms, load_rgb
from numpy.typing import NDArray

from ._precision import FloatProjection


class Case(TypedDict):
    id: str
    input: str
    focal_px: NotRequired[float]
    auto_focal: NotRequired[bool]
    exposure: NotRequired[float]


def encode(linear: NDArray[np.generic], exposure: float = 0) -> NDArray[np.float32]:
    exposed = linear.astype(np.float32) * np.float32(2.0**exposure)
    value = np.nan_to_num(exposed, nan=0, posinf=1, neginf=0).clip(0, 1).astype(np.float64)
    encoded = np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1 / 2.4) - 0.055)
    return encoded.astype(np.float32)


def load(path: Path, exposure: float = 0) -> NDArray[np.float32]:
    if path.suffix.lower() != ".exr":
        return load_rgb(str(path))[0].astype(np.float32) / np.float32(255)
    with OpenEXR.File(str(path), separate_channels=True) as file:
        rgb = np.stack([file.channels()[channel].pixels for channel in "RGB"], -1)
    return encode(rgb, exposure)


def infer(
    model: DepthPro2,
    case: Case,
    root: Path,
    repeat: int,
    half_head: bool,
    precision: str,
    out: Path,
) -> dict[str, object]:
    path = root / case["input"]
    auto_focal = case.get("auto_focal", False)
    rgb = load(path, case.get("exposure", 0))
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    inverse = np.empty((0,), dtype=np.float32)
    depth = np.empty((0,), dtype=np.float32)
    focal = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        image = torch.as_tensor(rgb.copy(), device="cuda").permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
        image = (image - mean) / std
        focal_arg = case.get("focal_px", 0.0)
        if auto_focal:
            estimator = model.fov
            if estimator is None:
                msg = "reference focal estimator missing"
                raise RuntimeError(msg)
            focal_arg = float(estimator.focal_length_px(image.unsqueeze(0)))
        with torch.autocast("cuda", dtype=torch.float16, enabled=half_head):
            prediction = model.infer(image, f_px=focal_arg)
        metric = prediction["depth"]
        focal_tensor = prediction["focallength_px"]
        if metric is None or focal_tensor is None:
            msg = "reference did not produce metric depth"
            raise RuntimeError(msg)
        inverse = prediction["inverse_depth"].float().cpu().numpy()
        depth = metric.float().cpu().numpy()
        focal = float(focal_tensor)
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    np.savez(
        out / (case["id"] + ".npz"),
        network_inverse_depth=inverse,
        camera_z_metres=depth,
        inverse_metres=np.float32(1) / depth,
        focal_px=focal,
        input_sha256=digest,
        half_head=half_head,
        precision=precision,
        exposure=case.get("exposure", 0),
        auto_focal=auto_focal,
        focal_input=case.get("focal_px", 0.0),
    )
    return {
        "id": case["id"],
        "frame_ms": times,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--stock-head", action="store_true")
    args = parser.parse_args()
    out: Path = args.out
    if args.repeat < 1:
        parser.error("repeat must be positive")
    cases = cast("list[Case]", json.loads(cast("Path", args.cases).read_text()))
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("cases must have nonempty, unique image IDs")
    start = time.perf_counter()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.precision]
    model, _ = create_model_and_transforms("depthpro2-h", precision=dtype)
    half_head = args.precision == "fp16" and not args.stock_head
    if half_head:
        model.head[-2] = FloatProjection(model.head[-2])
    torch.cuda.synchronize()
    session_ms = 1000 * (time.perf_counter() - start)
    out.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    for case in cases:
        row = infer(model, case, args.data_root, args.repeat, half_head, args.precision, out)
        row.update({"precision": args.precision, "session_ms": session_ms})
        report.append(row)
        (out / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
