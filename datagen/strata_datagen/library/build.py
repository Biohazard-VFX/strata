"""Prepare a build root's asset library from external sources (Megascans, HDRIs, hero library)."""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from strata_datagen import files
from strata_datagen.library import catalogue, coat, hdri, heroes, megascans, models, textures

BLENDER_SCRIPT = Path(__file__).resolve().parents[2] / "blender" / "fbx_to_usd.py"
EXCLUDED_PLANTS = ("houseplant", "succulent", "aloe", "palm", "paddle", "eelgrass", "basil", "ivy", "climber")
GRASS = ("grass", "wheat", "sedge", "rush", "reed")
ROCK = ("rock", "stone", "boulder")
DEBRIS = ("log", "stump", "branch", "debris", "root")
NATURAL_GROUND = ("ground", "soil", "grass", "moss", "sand", "gravel", "mud", "forest", "mulch", "rock")
MAN_MADE = ("trash", "manmade", "construction", "brick", "street", "building", "carved", "castle")


@dataclass(frozen=True)
class Sources:
    """External inputs, from a JSON file: megascans_roots, hdris, hero_library, heroes, max_variations."""

    megascans_roots: tuple[Path, ...]
    hdris: tuple[Path, ...]
    hero_library: Path
    heroes: tuple[str, ...]
    max_variations: int

    @staticmethod
    def load(path: Path) -> "Sources":
        raw = cast("dict[str, object]", json.loads(path.read_text()))
        return Sources(
            megascans_roots=tuple(Path(p) for p in cast("list[str]", raw["megascans_roots"])),
            hdris=tuple(Path(p) for p in cast("list[str]", raw["hdris"])),
            hero_library=Path(str(raw["hero_library"])),
            heroes=tuple(cast("list[str]", raw["heroes"])),
            max_variations=int(cast("int", raw["max_variations"])),
        )


def model_role(asset: megascans.Asset) -> str | None:
    name = asset.folder.name.lower()
    if asset.category == "3dplant":
        if any(word in name for word in EXCLUDED_PLANTS):
            return None
        if any(word in name for word in GRASS):
            return "grass"
        return "shrub" if "shrub" in name or (asset.max_size_m or 0.0) >= 1.0 else "groundcover"
    if any(word in name for word in MAN_MADE):
        return None
    if any(word in name for word in ROCK):
        return "rock"
    return "debris" if any(word in name for word in DEBRIS) else None


def natural_ground(asset: megascans.Asset) -> bool:
    name = asset.folder.name.lower()
    return any(word in name for word in NATURAL_GROUND) and not any(word in name for word in MAN_MADE)


def _assets(sources: Sources) -> Iterator[megascans.Asset]:
    seen: set[str] = set()
    for root in sources.megascans_roots:
        for category in ("3dplant", "3d", "surface"):
            for asset in megascans.scan(root, category):
                if asset.asset_id not in seen:
                    seen.add(asset.asset_id)
                    yield asset


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _textures(asset: megascans.Asset, folder: Path, root: Path) -> dict[str, str]:
    return {
        channel.lower(): _rel(textures.convert(found, folder), root)
        for channel, found in sorted(asset.maps.items())
    }


def _provenance(asset: megascans.Asset) -> catalogue.Provenance:
    return catalogue.Provenance(megascans.SOURCE, megascans.LICENSE, str(asset.folder))


def _tile_m(asset: megascans.Asset) -> float:
    return float(asset.max_size_m) if asset.max_size_m else 2.0


def _convert(jobs: list[tuple[Path, Path]], scratch: Path) -> None:
    if not jobs:
        return
    listing = scratch / "fbx_jobs.json"
    files.write_text(listing, json.dumps([[str(a), str(b)] for a, b in jobs]))
    subprocess.run(
        ["blender", "-b", "--factory-startup", "--python", str(BLENDER_SCRIPT), "--", str(listing)],
        check=True,
        capture_output=True,
        timeout=7200,
    )


def prepare_megascans(sources: Sources, root: Path) -> tuple[list[catalogue.Model], list[catalogue.Surface]]:
    scratch = root / "assets" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    surfaces: list[catalogue.Surface] = []
    pending: list[tuple[megascans.Asset, str, megascans.Model, Path]] = []
    for asset in _assets(sources):
        folder = root / "assets" / "megascans" / asset.asset_id
        if asset.category == "surface":
            if not natural_ground(asset):
                continue
            surfaces.append(
                catalogue.Surface(
                    asset.asset_id,
                    asset.name,
                    _tile_m(asset),
                    _textures(asset, folder / "textures", root),
                    _provenance(asset),
                )
            )
            continue
        role = model_role(asset)
        if role is None:
            continue
        pending.extend(
            (asset, role, model, scratch / f"{asset.asset_id}_v{model.variation}.usdc")
            for model in asset.models[: sources.max_variations]
        )
    _convert([(m.path, raw) for _, _, m, raw in pending if not raw.exists()], scratch)
    return _models(pending, root), surfaces


def _models(
    pending: list[tuple[megascans.Asset, str, megascans.Model, Path]], root: Path
) -> list[catalogue.Model]:
    out: list[catalogue.Model] = []
    maps: dict[str, dict[str, str]] = {}
    for asset, role, model, raw in pending:
        folder = root / "assets" / "megascans" / asset.asset_id
        if asset.asset_id not in maps:
            maps[asset.asset_id] = _textures(asset, folder / "textures", root)
        info = models.canonicalise(raw, folder / f"v{model.variation}.usdc")
        out.append(
            catalogue.Model(
                f"{asset.asset_id}_v{model.variation}",
                asset.asset_id,
                asset.name,
                role,
                _rel(info.file, root),
                info.size_m,
                info.triangles,
                maps[asset.asset_id],
                _provenance(asset),
            )
        )
    return out


def prepare_hdris(sources: Sources, root: Path) -> list[catalogue.Hdri]:
    out: list[catalogue.Hdri] = []
    for source in sources.hdris:
        key = source.stem.replace("_8k", "")
        target = root / "assets" / "hdri" / f"{key}.exr"
        cut = hdri.prepare(source, target)
        out.append(
            catalogue.Hdri(
                key,
                _rel(target, root),
                cut.direction,
                cut.irradiance,
                cut.color,
                cut.sky_irradiance,
                catalogue.Provenance(hdri.SOURCE, hdri.LICENSE, str(source)),
            )
        )
    return out


def prepare_heroes(sources: Sources, root: Path) -> list[catalogue.Hero]:
    out: list[catalogue.Hero] = []
    for key in sources.heroes:
        manifest = heroes.load(sources.hero_library / key)
        folder = root / "assets" / "heroes" / key
        prepare_pose = heroes.prepare_pose if coat.load(manifest.directory) is None else coat.prepare_pose
        poses = tuple(
            catalogue.HeroPose(pose, _rel(p.file, root), p.bounds_min, p.bounds_max, p.hairs)
            for pose in sorted(manifest.poses)
            for p in (prepare_pose(manifest, pose, folder / f"{pose}.usdc"),)
        )
        albedo = heroes.copy_albedo(manifest, folder / "textures")
        out.append(
            catalogue.Hero(
                key,
                _rel(albedo, root),
                poses,
                catalogue.Provenance(manifest.source, manifest.license, str(manifest.directory)),
            )
        )
    return out


def prepare(sources: Sources, root: Path) -> catalogue.Library:
    found_models, surfaces = prepare_megascans(sources, root)
    library = catalogue.Library(
        tuple(found_models),
        tuple(surfaces),
        tuple(prepare_hdris(sources, root)),
        tuple(prepare_heroes(sources, root)),
    )
    library.save(root)
    return library
