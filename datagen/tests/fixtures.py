"""Tiny rendered shots on disk, shaped like Karma output, for packing and QA tests."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenEXR
from numpy.typing import NDArray

from strata_datagen.exr import Deep, read_deep
from strata_datagen.render import pack
from strata_datagen.scene.export import CONTRACT_VERSION
from strata_dataset import identity, roles, sampling
from tests.flatten import flatten

FLIP_Z = np.diag([1.0, 1.0, -1.0, 1.0])
FOCAL_MM = 18.0
SENSOR_MM = 36.0


NAMES = {0: "sky", 1: "terrain", 2: "hero_skin", 3: "hero_groom", 4: "hero_hair"}


def membership(instance: int) -> tuple[float, float, float]:
    """The role primvars of a fixture id: the named fixed ids, any other id a foreground layer."""
    return roles.membership(NAMES.get(instance, "layer"))


@dataclass(frozen=True)
class Sample:
    """One deep sample: premultiplied grey, true float alpha (A is stored as half), id and depth.

    `checks` are the unpremultiplied `instance_check` and `instance_check2`, `roles` the
    unpremultiplied role channels; None stores those of the integer `instance`.
    """

    grey: float
    alpha: float
    instance: float
    z: float = 2.0
    checks: tuple[float, float] | None = None
    roles: tuple[float, float, float] | None = None

    def instance_checks(self) -> tuple[float, float]:
        if self.checks is not None:
            return self.checks
        ids = np.array([int(self.instance)])
        return float(identity.check(ids)[0]), float(identity.check2(ids)[0])

    def role_values(self) -> tuple[float, float, float]:
        return self.roles if self.roles is not None else membership(int(self.instance))


def merged(
    grey: float, alpha: float, sources: dict[int, float], z: float = 2.0, names: dict[int, str] | None = None
) -> Sample:
    """A sample Karma merged from `sources` (id to coverage share): every id and role channel is averaged.

    `names` gives the id range name of ids other than the fixture's named ones.
    """
    ids = np.array(list(sources))
    shares = np.array(list(sources.values()), np.float64)
    checks = tuple(
        float(np.dot(shares, h(ids).astype(np.float64))) for h in (identity.check, identity.check2)
    )
    named = {**NAMES, **(names or {})}
    members = [roles.membership(named.get(int(i), "layer")) for i in ids]
    mixture = np.dot(shares, np.array(members, np.float64))
    return Sample(
        grey,
        alpha,
        float(np.dot(shares, ids)),
        z,
        (checks[0], checks[1]),
        (float(mixture[0]), float(mixture[1]), float(mixture[2])),
    )


SKY = Sample(0.3, 1.0, 0, 3000.0)


def write_deep(path: Path, pixels: list[Sample | list[Sample]]) -> None:
    """A one-row deep EXR stored the way Karma writes it: half A and colour, float ids, coverage and roles."""
    rows = [[p] if isinstance(p, Sample) else p for p in pixels]
    values: dict[str, list[NDArray[np.float32]]] = {name: [] for name in ("R", "G", "B", "A", "Z", "ZBack")}
    values |= {
        name: [] for name in ("instance_id", "instance_check", "instance_check2", "coverage", *roles.ROLES)
    }
    for samples in rows:
        alpha = np.array([s.alpha for s in samples], np.float32)
        grey = np.array([s.grey for s in samples], np.float16).astype(np.float32)
        for c in "RGB":
            values[c].append(grey)
        values["A"].append(alpha.astype(np.float16).astype(np.float32))
        values["coverage"].append(alpha)
        values["instance_id"].append(np.array([s.instance for s in samples], np.float32) * alpha)
        for index, name in enumerate(("instance_check", "instance_check2")):
            values[name].append(np.array([s.instance_checks()[index] for s in samples], np.float32) * alpha)
        for index, name in enumerate(roles.ROLES):
            values[name].append(np.array([s.role_values()[index] for s in samples], np.float32) * alpha)
        values["Z"].append(np.array([s.z for s in samples], np.float32))
        values["ZBack"].append(values["Z"][-1].copy())
    channels: dict[str, OpenEXR.Channel] = {}
    for name, cells in values.items():
        grid = np.empty((1, len(rows)), dtype=object)
        for i, cell in enumerate(cells):
            grid[0, i] = cell
        channels[name] = OpenEXR.Channel(grid)
    header: dict[str, object] = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.deepscanline}
    path.parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File(header, channels).write(str(path))


def projection(width: int, height: int, fx: float, fy: float) -> NDArray[np.float64]:
    """Karma's camera-to-screen matrix (row vectors, +Z forward, screen window -1..1)."""
    out = np.zeros((4, 4))
    out[0, 0], out[1, 1] = 2.0 * fx / width, 2.0 * fy / height
    out[2, 2], out[2, 3], out[3, 2] = 1.0, 1.0, -0.03
    return out


def write_beauty(path: Path, deep: Deep, fy_scale: float = 1.0) -> None:
    """Beauty `C` equal to the flattened deep, a denoised copy, role AOVs and husk's camera header.

    The role AOVs are the deep's role coverage, as Karma renders them from the same primvars.
    """
    rgba = flatten(deep).rgba.astype(np.float32)
    channels = {c: rgba[..., i] for i, c in enumerate("RGBA")}
    channels |= {f"denoised.{c}": rgba[..., i] * np.float32(0.9) for i, c in enumerate("RGB")}
    frame = pack.samples(deep)
    covered = roles.coverage(frame.alpha, frame.roles, np.diff(frame.offsets), frame.shape)
    channels |= {aov: covered[c] for aov, c in zip(roles.AOVS, roles.CHANNELS, strict=True)}
    width, height = deep.width, deep.height
    fx = FOCAL_MM / SENSOR_MM * width
    world_to_camera = FLIP_Z.T
    header: dict[str, object] = {
        "compression": OpenEXR.ZIPS_COMPRESSION,
        "type": OpenEXR.scanlineimage,
        "name": "C",
        "worldToCamera_d": world_to_camera,
        "worldToNDC_d": world_to_camera @ projection(width, height, fx, fx * fy_scale),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File(header, {k: OpenEXR.Channel(np.ascontiguousarray(v)) for k, v in channels.items()}).write(
        str(path)
    )


def frame_record(frame: int, width: int, height: int) -> dict[str, object]:
    fx = FOCAL_MM / SENSOR_MM * width
    return {
        "frame": frame,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fx,
        "cx": width / 2,
        "cy": height / 2,
        "world_to_camera": np.eye(4).tolist(),
        "distortion": {"model": "none"},
        "shutter_open": 0.0,
        "shutter_close": 0.0,
        "fstop": 0.0,
        "focus_distance_m": 2.0,
        "focal_length_mm": FOCAL_MM,
        "sensor_width_mm": SENSOR_MM,
    }


def write_shot(shot: Path, frames: list[list[Sample | list[Sample]]]) -> None:
    """A shot of frames 1..N with raw deep on every frame; each frame is a row of pixels."""
    width = len(frames[0])
    records = [frame_record(i + 1, width, 1) for i in range(len(frames))]
    manifest = {
        "contract": CONTRACT_VERSION,
        "shot_id": shot.name,
        "seed": 0,
        "instance_ids": [
            {"name": "sky", "first_id": 0, "count": 1},
            {"name": "terrain", "first_id": 1, "count": 1},
            {"name": "hero_skin", "first_id": 2, "count": 1},
        ],
        "background": {"instance_id": 0, "radius_m": 3000.0, "far_clip_m": 5000.0},
        "render": {
            "frames": [1, len(frames)],
            "sampling_seed": sampling.schedule(0, range(1, len(frames) + 1)),
        },
        "scene": {
            "camera": {
                "move": "locked",
                "focal_length_mm": FOCAL_MM,
                "motion_blur": False,
                "depth_of_field": False,
            }
        },
    }
    shot.mkdir(parents=True, exist_ok=True)
    (shot / "manifest.json").write_text(json.dumps(manifest))
    (shot / "camera.json").write_text(json.dumps({"contract": CONTRACT_VERSION, "frames": records}))
    for index, pixels in enumerate(frames):
        name = f"{index + 1:04d}.exr"
        write_deep(shot / "deep" / name, pixels)
        write_beauty(shot / "beauty" / name, read_deep(shot / "deep" / name))
