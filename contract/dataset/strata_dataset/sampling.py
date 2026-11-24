"""Deterministic Karma sampling schedules shared by shot writers and record validation."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, TypeGuard, cast


class Schedule(TypedDict):
    mode: Literal["per-frame", "fixed"]
    seeds: list[int]


MASK = (1 << 31) - 1


def _key(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:render:{label}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & MASK


def _mix(value: int) -> int:
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & MASK
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & MASK
    return value ^ (value >> 16)


def schedule(shot_seed: int, frames: Sequence[int]) -> Schedule:
    """50/50 shot assignment; seeds are signed-int-safe and independent of task boundaries."""
    per_frame = bool(_key(shot_seed, "mode") & 1)
    base = _key(shot_seed, "seed")
    return {
        "mode": "per-frame" if per_frame else "fixed",
        "seeds": [_mix((base + (frame if per_frame else 0)) & MASK) for frame in frames],
    }


def noise_varies_per_frame(sampling: Mapping[str, object]) -> bool:
    return sampling["mode"] in {"per-frame", "karma_default"}


def _integers(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(type(item) is int for item in cast("list[object]", value))


def errors(manifest: Mapping[str, object]) -> list[str]:
    render = manifest.get("render")
    if not isinstance(render, dict):
        return ["missing render.sampling_seed"]
    render = cast("dict[str, object]", render)
    seed = manifest.get("seed")
    frames = render.get("frames")
    sampling = render.get("sampling_seed")
    if (
        type(seed) is not int
        or not _integers(frames)
        or len(frames) != 2
        or frames[0] > frames[1]
        or not isinstance(sampling, dict)
    ):
        return ["invalid render.sampling_seed inputs"]
    sampling = cast("dict[str, object]", sampling)
    if sampling.get("mode") == "karma_default":
        versions = manifest.get("generator_versions")
        if (
            sampling != {"mode": "karma_default"}
            or not isinstance(versions, dict)
            or cast("dict[str, object]", versions).get("settings") != "render-settings-0.4.0"
        ):
            return ["render.sampling_seed karma_default requires unauthored render-settings-0.4.0 evidence"]
        return []
    seeds = sampling.get("seeds")
    if (
        not _integers(seeds)
        or len(seeds) != frames[1] - frames[0] + 1
        or sampling != schedule(seed, range(frames[0], frames[1] + 1))
    ):
        return ["render.sampling_seed differs from the shot seed schedule"]
    return []
