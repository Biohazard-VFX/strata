"""Validate shot directories against the Strata dataset contract.

Usage: python validate.py SHOT_DIR [SHOT_DIR ...]   (exit status 1 when any shot is invalid)
"""

import json
import sys
from pathlib import Path
from typing import cast

import OpenEXR
from jsonschema import Draft202012Validator

SCHEMA = cast("dict[str, object]", json.loads((Path(__file__).parent / "schema.json").read_text()))
BEAUTY = {
    "C": {"R", "G", "B", "A"},
    "albedo": {"albedo.R", "albedo.G", "albedo.B"},
    "N": {"N.x", "N.y", "N.z"},
    "roughness": {"roughness.roughness"},
    "metallic": {"metallic.metallic"},
    "diffuse_direct": {"diffuse_direct.R", "diffuse_direct.G", "diffuse_direct.B"},
    "diffuse_indirect": {"diffuse_indirect.R", "diffuse_indirect.G", "diffuse_indirect.B"},
    "specular_direct": {"specular_direct.R", "specular_direct.G", "specular_direct.B"},
    "specular_indirect": {"specular_indirect.R", "specular_indirect.G", "specular_indirect.B"},
    "Z": {"Z.Z"},
    "instance_id": {"instance_id.instance_id"},
}
DEEP = {"R", "G", "B", "A", "Z", "ZBack", "instance_id"}


def _schema_errors(record: dict[str, object], definition: str) -> list[str]:
    schema = {**SCHEMA, "$ref": f"#/$defs/{definition}"}
    errors = list(Draft202012Validator(schema).iter_errors(record))
    return [f"{definition}: {'/'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in errors]


def _channels(header: dict[str, object]) -> set[str]:
    return {str(getattr(channel, "name", channel)) for channel in cast("list[object]", header["channels"])}


def _size(header: dict[str, object]) -> tuple[int, int]:
    low, high = cast("tuple[list[int], list[int]]", header["dataWindow"])
    return int(high[0]) - int(low[0]) + 1, int(high[1]) - int(low[1]) + 1


def _beauty_errors(path: Path, size: tuple[int, int]) -> list[str]:
    parts = {p.name(): p for p in OpenEXR.File(str(path), header_only=True).parts}
    errors = [f"{path.name}: missing part {name}" for name in BEAUTY if name not in parts]
    for name, channels in BEAUTY.items():
        if name in parts and not channels <= _channels(parts[name].header):
            errors.append(f"{path.name}: part {name} lacks {sorted(channels)}")
    if "C" in parts and _size(parts["C"].header) != size:
        errors.append(f"{path.name}: resolution differs from camera.json")
    return errors


def _deep_errors(path: Path, size: tuple[int, int]) -> list[str]:
    part = OpenEXR.File(str(path), header_only=True).parts[0]
    errors: list[str] = []
    if str(part.header.get("type")) != "Storage.deepscanline":
        errors.append(f"{path.name}: not a deep scanline image")
    missing = DEEP - _channels(part.header)
    if missing:
        errors.append(f"{path.name}: deep image lacks {sorted(missing)}")
    if _size(part.header) != size:
        errors.append(f"{path.name}: deep resolution differs from camera.json")
    return errors


def validate(shot: Path) -> list[str]:
    try:
        manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
        camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable records: {error}"]
    errors = _schema_errors(manifest, "manifest") + _schema_errors(camera, "camera")
    if errors:
        return errors
    for record in cast("list[dict[str, int]]", camera["frames"]):
        name = f"{record['frame']:04d}.exr"
        size = (record["width"], record["height"])
        for folder, check in (("beauty", _beauty_errors), ("deep", _deep_errors)):
            path = shot / folder / name
            errors += check(path, size) if path.is_file() else [f"missing {folder}/{name}"]
    return errors


def main() -> None:
    failed = False
    for argument in sys.argv[1:]:
        errors = validate(Path(argument))
        failed |= bool(errors)
        print(f"{argument}: {'ok' if not errors else 'INVALID'}")
        for error in errors:
            print(f"  {error}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
