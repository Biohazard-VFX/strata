"""Measure serial CUDA/TensorRT/PyTorch runs and sample process VRAM with nvidia-smi."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

from PIL import Image


def gpu_memory() -> tuple[int, int, dict[int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used, free = (int(value.strip()) for value in output.strip().split(","))
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes: dict[int, int] = {}
    for line in output.splitlines():
        if line.strip():
            pid, memory = (int(value.strip()) for value in line.split(","))
            processes[pid] = memory
    return used, free, processes


def measured(command: list[str], prefix: Path, timeout: float) -> dict[str, object]:
    baseline, free, _ = gpu_memory()
    print(
        json.dumps({"claim": str(prefix), "device_used_mib": baseline, "device_free_mib": free}),
        flush=True,
    )
    if free < 14 * 1024:
        msg = "benchmark requires 14 GiB free before claiming the shared GPU"
        raise RuntimeError(msg)
    device_peak, process_peak, samples = baseline, 0, 0
    start = time.monotonic()
    timed_out = False
    with (
        prefix.with_suffix(".stdout.log").open("w") as stdout,
        prefix.with_suffix(".stderr.log").open("w") as stderr,
        subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env={**os.environ, "PYTHONPATH": ""}
        ) as process,
    ):
        while process.poll() is None:
            used, _, processes = gpu_memory()
            device_peak = max(device_peak, used)
            process_peak = max(process_peak, processes.get(process.pid, 0))
            samples += 1
            if time.monotonic() - start > timeout:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(0.1)
        returncode = process.wait()
    print(json.dumps({"released": str(prefix), "returncode": returncode}), flush=True)
    result: dict[str, object] = {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "device_baseline_mib": baseline,
        "device_peak_mib": device_peak,
        "process_peak_mib": process_peak,
        "vram_samples": samples,
        "wall_ms": 1000 * (time.monotonic() - start),
    }
    if returncode:
        result["error"] = "\n".join(
            prefix.with_suffix(".stderr.log").read_text().splitlines()[-15:]
        )
        return result
    rows = [
        cast("dict[str, object]", json.loads(line))
        for line in prefix.with_suffix(".stdout.log").read_text().splitlines()
        if line.startswith("{")
    ]
    if len(rows) != 1 or process_peak == 0:
        msg = "missing timing JSON or process VRAM samples"
        raise RuntimeError(msg)
    timing = rows[0]
    frames = cast("list[float]", timing["frame_ms"])
    result.update(timing)
    result.update(first_frame_ms=frames[0], warm_median_ms=statistics.median(frames[1:]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("cuda", "tensorrt", "torch", "torch-stock"),
        default=["cuda", "torch"],
    )
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("repeat must include a first frame and at least one warm frame")
    out = cast("Path", args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    for width, height, focal in ((1920, 1080, 2666.67), (3840, 2160, 5333.34)):
        image = out / f"input-{width}x{height}.png"
        with Image.open(args.image) as source:
            source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(image)
        for engine in cast("list[str]", args.engines):
            prefix = out / f"{engine}-{width}x{height}"
            if engine.startswith("torch"):
                manifest = prefix.with_suffix(".json")
                manifest.write_text(
                    json.dumps([{"id": prefix.name, "input": str(image), "focal_px": focal}]) + "\n"
                )
                command = [
                    sys.executable,
                    "-m",
                    "strata_train.reference",
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(prefix),
                    "--repeat",
                    str(args.repeat),
                ]
                if engine == "torch-stock":
                    command.extend(["--precision", "bf16", "--stock-head"])
            else:
                command = [
                    str(args.cli.resolve()),
                    "depth",
                    "--model",
                    str(args.model.resolve()),
                    "--in",
                    str(image),
                    "--out-depth",
                    str(prefix.with_suffix(".depth.exr")),
                    "--out-deep",
                    str(prefix.with_suffix(".deep.exr")),
                    "--f-px",
                    str(focal),
                    "--provider",
                    engine,
                    "--repeat",
                    str(args.repeat),
                ]
            result = measured(command, prefix, args.timeout)
            result.update(engine=engine, width=width, height=height)
            report.append(result)
            (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
