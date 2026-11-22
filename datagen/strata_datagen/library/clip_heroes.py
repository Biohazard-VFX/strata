"""Animated heroes from the external hero library: hero.json with `clips` instead of static poses.

Animated hero.json fields (written by library/humans.py; see datagen/README.md for the full schema):
  key, kind; rest: rest-pose USD; clips: {name: {file, category, frames, first_frame, fps, subframes,
  speed_mps, bounds_min, bounds_max, track, source}}; skin_prim; parts; regions (names of the skin's
  `region_<name>` vertex primvars, strata_datagen.regions); forward; facial_hair; rest_primvar;
  source, license, licenses, redistributable. A scalp region gives the hero a hairstyle per shot,
  bound to the `rest_primvar` points and carried by the animated skin.
Clip files carry their own MaterialX looks, so a build root copies the hero's usd/ and textures/.
"""

import json
import shutil
from pathlib import Path
from typing import cast

from pxr import Usd, UsdGeom

from strata_datagen import regions as skin_regions
from strata_datagen.library import catalogue, heroes

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
    skin_prim = str(raw["skin_prim"])
    regions = tuple(str(r) for r in cast("list[object]", raw["regions"]))
    rest = Usd.Stage.Open(str(folder / str(raw["rest"])))
    skin_regions.read(UsdGeom.Mesh(rest.GetPrimAtPath(skin_prim)), regions)
    return catalogue.Hero(
        str(raw["key"]),
        "",
        poses,
        catalogue.Provenance(str(raw["source"]), str(raw["license"]), str(directory)),
        heroes.hair(
            regions,
            _vec(raw["forward"]),
            facial_hair=bool(raw["facial_hair"]),
            skin_prim=skin_prim,
            rest_primvar=str(raw["rest_primvar"]),
        ),
    )
