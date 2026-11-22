"""Animated heroes from the external hero library: hero.json with `clips` instead of static poses.

Animated hero.json fields (written by library/humans.py; see datagen/README.md for the full schema):
  key, kind; rest: rest-pose USD; clips: {name: {file, category, frames, first_frame, fps, subframes,
  speed_mps, bounds_min, bounds_max, track, source}}; skin_prim; parts; regions (GeomSubset paths on
  the skin, family `region_family`); rest_primvar; source, license, licenses, redistributable.
Clip files carry their own MaterialX looks, so a build root copies the hero's usd/ and textures/.
"""

import json
import shutil
from pathlib import Path
from typing import cast

from strata_datagen.library import catalogue

COPIED = ("usd", "textures")


def is_animated(directory: Path) -> bool:
    return "clips" in cast("dict[str, object]", json.loads((directory / "hero.json").read_text()))


def _vec(raw: object) -> tuple[float, float, float]:
    x, y, z = cast("list[float]", raw)
    return (float(x), float(y), float(z))


def prepare(directory: Path, folder: Path, root: Path) -> catalogue.Hero:
    raw = cast("dict[str, object]", json.loads((directory / "hero.json").read_text()))
    for name in COPIED:
        target = folder / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(directory / name, target)
    clips = cast("dict[str, dict[str, object]]", raw["clips"])
    poses = tuple(
        catalogue.HeroPose(
            name,
            (folder / str(clip["file"])).relative_to(root).as_posix(),
            _vec(clip["bounds_min"]),
            _vec(clip["bounds_max"]),
            0,
            frames=int(cast("int", clip["frames"])),
            track=tuple(_vec(row) for row in cast("list[object]", clip["track"])),
        )
        for name, clip in sorted(clips.items())
    )
    return catalogue.Hero(
        str(raw["key"]),
        "",
        poses,
        catalogue.Provenance(str(raw["source"]), str(raw["license"]), str(directory)),
    )
