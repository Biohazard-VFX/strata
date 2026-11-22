"""Material looks for library entries, authored once per asset under /World/materials."""

from pathlib import Path

import numpy as np
from pxr import Usd, UsdShade

from strata_datagen.generators import materials, thin
from strata_datagen.library.catalogue import Hero, Model, Surface
from strata_datagen.usdutil import relative_asset

MATERIALS = "/World/materials"


def _texture(root: Path, scene_dir: Path, textures: dict[str, str], channel: str) -> str | None:
    file = textures.get(channel)
    return None if file is None else relative_asset(root / file, scene_dir)


def _textures(root: Path, scene_dir: Path, textures: dict[str, str], *, cutout: bool) -> materials.Textures:
    albedo = _texture(root, scene_dir, textures, "albedo")
    if albedo is None:
        raise ValueError("an albedo map is required")
    return materials.Textures(
        albedo=albedo,
        roughness=_texture(root, scene_dir, textures, "roughness"),
        normal=_texture(root, scene_dir, textures, "normal"),
        opacity=_texture(root, scene_dir, textures, "opacity") if cutout else None,
    )


def safe_name(key: str) -> str:
    return "m_" + "".join(c if c.isalnum() else "_" for c in key)


def model(stage: Usd.Stage, root: Path, scene_dir: Path, entry: Model, rng: np.random.Generator) -> str:
    path = f"{MATERIALS}/{safe_name(entry.asset_id)}"
    if stage.GetPrimAtPath(path):
        return path
    plant = entry.role in ("grass", "groundcover", "shrub")
    tint = tuple(float(v) for v in 1.0 + rng.uniform(-0.08, 0.08, 3)) if plant else (1.0, 1.0, 1.0)
    look = materials.Look(
        _textures(root, scene_dir, entry.textures, cutout=True),
        foliage=plant,
        tint=(tint[0], tint[1], tint[2]),
    )
    materials.author(stage, path, look)
    return path


def ground(stage: Usd.Stage, root: Path, scene_dir: Path, entry: Surface) -> UsdShade.Material:
    look = materials.Look(
        _textures(root, scene_dir, entry.textures, cutout=False), tile_m=entry.tile_m, macro_variation=0.25
    )
    return materials.author(stage, f"{MATERIALS}/ground", look)


def hero(
    stage: Usd.Stage, root: Path, scene_dir: Path, entry: Hero
) -> tuple[UsdShade.Material, UsdShade.Material]:
    albedo = relative_asset(root / entry.albedo, scene_dir)
    skin = materials.author(
        stage, f"{MATERIALS}/hero_skin", materials.Look(materials.Textures(albedo), roughness=0.6)
    )
    groom = materials.author(
        stage, f"{MATERIALS}/hero_groom", materials.Look(materials.Textures(albedo), roughness=0.45)
    )
    return skin, groom


def thin_finish(stage: Usd.Stage, finish: thin.Finish) -> UsdShade.Material:
    path = f"{MATERIALS}/thin_{finish.name}"
    if stage.GetPrimAtPath(path):
        return UsdShade.Material(stage.GetPrimAtPath(path))
    look = materials.Look(None, roughness=finish.roughness, base_color=finish.color, metallic=finish.metallic)
    return materials.author(stage, path, look)
