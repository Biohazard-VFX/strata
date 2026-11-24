"""Per-shot records of the dataset contract: camera.json and manifest.json.

The manifest is written only for a composed scene whose rendered surfaces carry the id and role
primvars of its declared id ranges (`scene.audit`), so a shot with wrong tags is never rendered.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pxr import Usd

from strata_datagen import files
from strata_datagen.generators import (
    camera,
    groom,
    hairstyle,
    hero,
    human,
    lighting,
    materials,
    scatter,
    terrain,
    thin,
)
from strata_datagen.library.catalogue import Library, Provenance
from strata_datagen.render import settings
from strata_datagen.scene import audit
from strata_datagen.scene.compose import FIRST_SCATTER_ID, TERRAIN_ID, RenderParams, Shot, samples_for
from strata_datagen.scene.spec import RECIPE_VERSION
from strata_dataset import records, sampling

CONTRACT_VERSION = records.VERSION
GENERATORS = {
    m.__name__.rsplit(".", 1)[-1]: m.VERSION
    for m in (terrain, scatter, materials, lighting, camera, hero, human, groom, hairstyle, thin, settings)
}


def _rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in matrix]


def camera_record(shot: Shot, render: RenderParams) -> dict[str, object]:
    track = shot.track
    frames = [
        {
            "frame": frame,
            "width": render.width,
            "height": render.height,
            "fx": float(k[0]),
            "fy": float(k[1]),
            "cx": float(k[2]),
            "cy": float(k[3]),
            "world_to_camera": _rows(w2c),
            "distortion": {"model": "none"},
            "shutter_open": track.shutter[0],
            "shutter_close": track.shutter[1],
            "fstop": float(lens[1]),
            "focus_distance_m": float(lens[2]),
            "focal_length_mm": float(lens[0]),
            "sensor_width_mm": camera.SENSOR_MM[0],
        }
        for frame, w2c, k, lens in zip(
            track.frames,
            track.world_to_camera(),
            track.intrinsics(render.width, render.height),
            track.frame_lenses(),
            strict=True,
        )
    ]
    return {
        "contract": CONTRACT_VERSION,
        "convention": "camera looks down -Z, +Y up, column vectors",
        "move": track.move,
        "frames": frames,
    }


def _asset(key: str, provenance: Provenance) -> dict[str, str]:
    return {
        "id": key,
        "source": provenance.source,
        "license": provenance.license,
        "origin": provenance.origin,
    }


def _assets(shot: Shot, library: Library) -> list[dict[str, str]]:
    surface = next(s for s in library.surfaces if s.asset_id == shot.surface)
    hdri = next(h for h in library.hdris if h.key == shot.light.hdri.key)
    hero_entry = library.hero(shot.hero_key)
    found = [
        _asset(surface.asset_id, surface.provenance),
        _asset(hdri.key, hdri.provenance),
        _asset(f"{hero_entry.key}:{shot.pose.pose}", hero_entry.provenance),
    ]
    found += [_asset(m.asset_id, m.provenance) for m in shot.assets.values()]
    unique = {a["id"]: a for a in found}
    return [unique[k] for k in sorted(unique)]


def _ids(shot: Shot) -> list[dict[str, object]]:
    fixed: list[dict[str, object]] = [
        {"name": "sky", "first_id": lighting.SKY_ID, "count": 1},
        {"name": "terrain", "first_id": TERRAIN_ID, "count": 1},
        {"name": "hero_skin", "first_id": hero.SKIN_ID, "count": 1},
    ]
    if shot.groomed:
        fixed.append({"name": "hero_groom", "first_id": hero.GROOM_ID, "count": 1})
    if shot.hair is not None:
        fixed.append({"name": "hero_hair", "first_id": hero.HAIR_ID, "count": 1})
    return fixed + shot.layers


def manifest(shot: Shot, library: Library, render: RenderParams) -> dict[str, object]:
    spec = asdict(shot.spec)
    return {
        "contract": CONTRACT_VERSION,
        "shot_id": shot.spec.shot_id,
        "seed": shot.spec.seed,
        "recipe_version": RECIPE_VERSION,
        "spec": spec,
        "generator_versions": GENERATORS,
        "scene": {
            "terrain": shot.terrain.as_dict(),
            "ground_surface": shot.surface,
            "hero": {
                "key": shot.hero_key,
                "pose": shot.pose.pose,
                "speed_mps": shot.motion.speed_mps,
                "hairs": shot.pose.hairs,
                "hairstyle": None if shot.hair is None else hairstyle.record(shot.hair),
                "clip_start": None if shot.placement is None else shot.placement.start,
                "animation": shot.hero_animation,
            },
            "lighting": {
                "hdri": shot.light.hdri.key,
                "rotation_deg": shot.light.rotation_deg,
                "exposure_scale": shot.light.exposure_scale,
            },
            "camera": {
                "move": shot.track.move,
                "focal_length_mm": shot.track.lens.focal_mm,
                "fstop": shot.track.lens.fstop,
                "focus_distance_m": shot.track.lens.focus_m,
                "motion_blur": shot.track.motion_blur,
                "depth_of_field": shot.track.depth_of_field,
            },
            "thin": [s.record() for s in shot.thin],
        },
        "instance_ids": _ids(shot),
        "background": {
            "instance_id": lighting.SKY_ID,
            "radius_m": lighting.BACKDROP_M,
            "far_clip_m": camera.CLIPPING_M[1],
        },
        "first_scatter_id": FIRST_SCATTER_ID,
        "assets": _assets(shot, library),
        "render": {
            "renderer": "Karma XPU",
            "width": render.width,
            "height": render.height,
            "path_traced_samples": samples_for(render.path_traced_samples, shot.track),
            "frames": [shot.frames[0], shot.frames[-1]],
            "aovs": [a.name for a in settings.BEAUTY_AOVS],
            "deep_aovs": list(settings.DEEP_AOVS),
            "sampling_seed": sampling.schedule(shot.spec.seed, shot.frames),
        },
        "timings": {},
    }


def write(shot: Shot, library: Library, render: RenderParams) -> None:
    """camera.json and manifest.json; raises ValueError, writing neither, if the scene fails its tag audit."""
    errors = audit.tag_errors(Usd.Stage.Open(str(shot.scene_dir / "shot.usda")), _ids(shot))
    if errors:
        raise ValueError(f"{shot.spec.shot_id}: {'; '.join(errors[:8])}")
    shot.directory.mkdir(parents=True, exist_ok=True)
    for name, record in (
        ("camera.json", camera_record(shot, render)),
        ("manifest.json", manifest(shot, library, render)),
    ):
        files.write_text(shot.directory / name, json.dumps(record, indent=1, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(path.read_text())
    return loaded
