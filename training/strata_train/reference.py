"""Generate independent PyTorch oracle arrays without sharing a GPU with ORT."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np
import OpenEXR
import torch
from depth_pro_2 import DepthPro2, create_model_and_transforms, load_rgb
from numpy.typing import NDArray
from PIL import Image

from ._external import resolve
from ._precision import FloatProjection


class Case(TypedDict):
    id: str
    input: str
    focal_px: float
    auto_focal: NotRequired[bool]
    exposure: NotRequired[float]


def load(path: Path, exposure: float = 0) -> NDArray[np.float32]:
    if path.suffix.lower() != ".exr":
        return load_rgb(str(path))[0].astype(np.float32) / np.float32(255)
    with OpenEXR.File(str(path), separate_channels=True) as file:
        rgb = np.stack([file.channels()[channel].pixels for channel in "RGB"], -1)
    value = (rgb.astype(np.float32) * np.float32(2**exposure)).clip(0, 1)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, np.float32(1 / 2.4)) - 0.055,
    ).astype(np.float32)


def dataset(root: Path) -> list[Case]:
    images = [root / "results/v001/beauty.png"]
    images.extend(sorted((root / "results").glob("v003_*/beauty.png")))
    images.extend(sorted((root / "renders/v004_dataset").glob("*/beauty*.exr")))
    return [
        Case(id=path.parent.name, input=str(path.resolve()), focal_px=2666.67) for path in images
    ]


def validation_cases(cases: list[Case], out: Path) -> list[Case]:
    out.mkdir(parents=True, exist_ok=True)
    result: list[Case] = []
    with Image.open(cases[0]["input"]) as source:
        for name, size in (
            ("portrait", (257, 385)),
            ("landscape", (451, 251)),
            ("panorama", (769, 128)),
        ):
            path = out / (name + ".png")
            source.resize(size, Image.Resampling.LANCZOS).save(path)
            result.append(
                Case(id="auto_" + name, input=str(path.resolve()), focal_px=0, auto_focal=True)
            )
    exrs = [case for case in cases if Path(case["input"]).suffix.lower() == ".exr"]
    if exrs:
        case = next((case for case in exrs if case["id"] == "sharp_06"), exrs[0])
        result.append(
            Case(id="exposure_minus2", input=case["input"], focal_px=case["focal_px"], exposure=-2)
        )
    return result


def infer(
    model: DepthPro2,
    case: Case,
    repeat: int,
    auto_focal: bool,
    half_head: bool,
    out: Path,
) -> dict[str, object]:
    rgb = load(Path(case["input"]), case.get("exposure", 0))
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
        focal_arg = case["focal_px"]
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
    with Path(case["input"]).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    np.savez(
        out / (case["id"] + ".npz"),
        network_inverse_depth=inverse,
        camera_z_metres=depth,
        inverse_metres=np.float32(1) / depth,
        focal_px=focal,
        input_sha256=digest,
        half_head=half_head,
        exposure=case.get("exposure", 0),
        auto_focal=auto_focal,
        focal_input=0 if auto_focal else case["focal_px"],
    )
    return {
        "id": case["id"],
        "frame_ms": times,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--auto-focal", action="store_true")
    parser.add_argument("--stock-head", action="store_true")
    parser.add_argument("--validation-cases", action="store_true")
    args = parser.parse_args()
    manifest: Path = args.manifest
    out: Path = args.out
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.data_root:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(dataset(Path(args.data_root)), indent=2) + "\n")
    cases = cast("list[Case]", json.loads(manifest.read_text()))
    if args.validation_cases and cases:
        cases.extend(validation_cases(cases, out / "inputs"))
        manifest.write_text(json.dumps(cases, indent=2) + "\n")
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("manifest must have nonempty, unique image IDs")
    torch.set_num_threads(4)
    limit = cast("Callable[[float], None]", resolve(torch.cuda, "set_per_process_memory_fraction"))
    limit(13 * 2**30 / torch.cuda.mem_get_info()[1])
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.precision]
    start = time.perf_counter()
    model, _ = create_model_and_transforms("depthpro2-h", precision=dtype)
    if args.precision == "fp16" and not args.stock_head:
        model.head[-2] = FloatProjection(model.head[-2])
    torch.cuda.synchronize()
    session_ms = 1000 * (time.perf_counter() - start)
    out.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    for case in cases:
        row = infer(
            model,
            case,
            args.repeat,
            args.auto_focal or case.get("auto_focal", False),
            args.precision == "fp16" and not args.stock_head,
            out,
        )
        row.update({"precision": args.precision, "session_ms": session_ms})
        report.append(row)
        (out / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
