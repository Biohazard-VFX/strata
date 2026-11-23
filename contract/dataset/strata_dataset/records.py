"""The contract version every shot's `manifest.json` and `camera.json` carries, and what it implies.

Readers and writers refuse records of any other version instead of reinterpreting them.
"""

from collections.abc import Mapping, Sequence
from typing import cast

from strata_dataset import identity, sampling

VERSION = "strata-dataset-0.4.0"


def errors(manifest: Mapping[str, object], camera: Mapping[str, object]) -> list[str]:
    """Why these records are not this contract version or declare undecodable ids; empty when they are."""
    found = [
        f"{name}.json is {record.get('contract')!r}, not {VERSION!r}"
        for name, record in (("manifest", manifest), ("camera", camera))
        if record.get("contract") != VERSION
    ]
    for record in cast("Sequence[Mapping[str, int]]", manifest.get("instance_ids", [])):
        found += identity.range_errors(record["first_id"], record["count"])
    return found + sampling.errors(manifest)
