"""The contract version every shot's `manifest.json` and `camera.json` carries, and what it implies.

Readers and writers refuse records of any other version instead of reinterpreting them.
"""

from collections.abc import Mapping, Sequence
from typing import cast

from strata_dataset import identity, sampling

VERSION = "strata-dataset-0.5.0"


def errors(manifest: Mapping[str, object], camera: Mapping[str, object]) -> list[str]:
    """Why these records are not this contract version or declare undecodable ids; empty when they are."""
    found = [
        f"{name}.json is {record.get('contract')!r}, not {VERSION!r}"
        for name, record in (("manifest", manifest), ("camera", camera))
        if record.get("contract") != VERSION
    ]
    found += sampling.errors(manifest)
    if found:
        return found
    render = cast("Mapping[str, object]", manifest["render"])
    if "deep_stride" in render:
        found.append("render.deep_stride is forbidden: every frame requires deep")
    first, last = cast("Sequence[int]", render["frames"])
    frames = cast("Sequence[Mapping[str, object]]", camera["frames"])
    if [frame["frame"] for frame in frames] != list(range(first, last + 1)):
        found.append(f"camera.json frames do not cover the rendered range {first}..{last}")
    for record in cast("Sequence[Mapping[str, int]]", manifest.get("instance_ids", [])):
        found += identity.range_errors(record["first_id"], record["count"])
    return found
