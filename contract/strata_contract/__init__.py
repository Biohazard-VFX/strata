"""Versioned model contract shared by export and the C runtime."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import cast

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


def spec() -> dict[str, Json]:
    """Read the canonical version-one model interface."""
    return cast(
        "dict[str, Json]", json.loads(files(__package__).joinpath("model_v1.json").read_text())
    )


def write_manifest(model: Path) -> Path:
    """Write the exact runtime contract beside an exported model."""
    path = model.with_suffix(".json")
    path.write_text(json.dumps(spec(), indent=2) + "\n")
    return path


def validate_manifest(path: Path) -> None:
    """Reject incompatible names, constants, shapes, or semantics."""
    if json.loads(path.read_text()) != spec():
        msg = f"incompatible Strata model contract: {path}"
        raise ValueError(msg)


def c_header() -> str:
    """Generate C constants from the same JSON used by the exporter."""
    data = spec()
    preprocess = cast("dict[str, Json]", data["preprocessing"])
    fov = cast("dict[str, Json]", data["fov"])
    image = cast("dict[str, Json]", data["input"])
    output = cast("dict[str, Json]", data["output"])
    mean = cast("list[float]", preprocess["mean"])
    std = cast("list[float]", preprocess["std"])
    constants: dict[str, Json] = {
        "INPUT_NAME": image["name"],
        "OUTPUT_NAME": output["name"],
        "FOV_OUTPUT_NAME": fov["output_name"],
        "FOV_SUFFIX": fov["suffix"],
        "INPUT_MULTIPLE": preprocess["multiple"],
        "MAX_HEIGHT": preprocess["max_height"],
        "MAX_WIDTH": preprocess["max_width"],
        "FOV_SHORT_MAX": fov["short_side_max"],
        "FOV_ASPECT_MAX": fov["aspect_max"],
    }
    for i, channel in enumerate("RGB"):
        constants[f"MEAN_{channel}"] = mean[i]
        constants[f"STD_{channel}"] = std[i]
    for key in ("threshold", "linear_scale", "power_scale", "power_offset", "gamma"):
        constants[f"SRGB_{key.upper()}"] = preprocess[f"srgb_{key}"]
    lines = ["#ifndef STRATA_CONTRACT_H", "#define STRATA_CONTRACT_H", ""]
    for key, value in constants.items():
        literal = json.dumps(value)
        if isinstance(value, float):
            literal += "f"
        lines.append(f"#define STRATA_{key} {literal}")
    canonical = json.dumps(data, separators=(",", ":"))
    lines.extend(["", f"#define STRATA_CONTRACT_JSON {json.dumps(canonical)}", "", "#endif", ""])
    return "\n".join(lines)
