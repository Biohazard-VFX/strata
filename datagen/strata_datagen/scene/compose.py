"""Compose one shot: terrain, hero, camera, scatter, lights and render settings into USD layers."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.fields import Ground
from strata_datagen.generators import camera, hairstyle, hero, lighting, materials, scatter, terrain
from strata_datagen.generators.scatter import Disk, Layer
from strata_datagen.library import heroes
from strata_datagen.library.catalogue import Hero, HeroPose, Library, Model
from strata_datagen.render import settings
from strata_datagen.render.pack import DEEP_STRIDE
from strata_datagen.rng import stream
from strata_datagen.scene import layers, looks
from strata_datagen.scene.spec import DEFAULT_FRAMES, ShotSpec, frames
from strata_datagen.usdutil import new_stage, relative_asset, tag_instances

TERRAIN_ID = 1
FIRST_SCATTER_ID = 1000
TARGET_IRRADIANCE = 3.0
GROWING = ("grass_near", "grass_far", "groundcover", "shrubs")


@dataclass(frozen=True)
class RenderParams:
    width: int
    height: int
    path_traced_samples: int
    deep_stride: int = DEEP_STRIDE


@dataclass
class Shot:
    """Everything sampled for a shot, plus the records its manifest needs."""

    spec: ShotSpec
    directory: Path
    ground: Ground
    terrain: terrain.TerrainParams
    surface: str
    hero_key: str
    pose: HeroPose
    motion: hero.HeroMotion
    track: camera.CameraTrack
    frames: tuple[int, ...]
    light: lighting.Lighting
    layers: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    assets: dict[str, Model] = field(default_factory=dict[str, Model])
    hair: hairstyle.Style | None = None

    @property
    def scene_dir(self) -> Path:
        return self.directory / "scene"


def _choice(rng: np.random.Generator, options: list[str], fixed: str | None) -> str:
    drawn = options[int(rng.integers(0, len(options)))]
    return drawn if fixed is None else fixed


def _hero_pose(library: Library, spec: ShotSpec, rng: np.random.Generator) -> tuple[Hero, HeroPose]:
    entry = library.hero(_choice(rng, sorted(h.key for h in library.heroes), spec.hero))
    pose = _choice(rng, sorted(p.pose for p in entry.poses), spec.pose)
    return entry, next(p for p in entry.poses if p.pose == pose)


def _hdris(library: Library, root: Path, scene_dir: Path) -> tuple[lighting.Hdri, ...]:
    return tuple(
        lighting.Hdri(
            h.key,
            relative_asset(root / h.file, scene_dir),
            h.sun_direction,
            h.sun_irradiance,
            h.sun_color,
            h.sky_irradiance,
        )
        for h in library.hdris
    )


def plan(spec: ShotSpec, library: Library, root: Path, frame_count: int = DEFAULT_FRAMES) -> Shot:
    directory = root / "shots" / spec.shot_id
    clip = frames(spec, frame_count)
    params = terrain.sample_params(stream(spec.seed, "terrain"))
    ground = terrain.build(params, stream(spec.seed, "terrain", "noise"))
    surface = _choice(stream(spec.seed, "ground"), sorted(s.asset_id for s in library.surfaces), spec.ground)
    entry, pose = _hero_pose(library, spec, stream(spec.seed, "hero"))
    samples = len(clip) + 2
    motion = hero.sample_motion(
        stream(spec.seed, "hero", "motion"), ground, samples, -pose.bounds_min[1], spec.hero_speed_mps
    )
    height = pose.bounds_max[1] - pose.bounds_min[1]
    subject = max(pose.bounds_max[0] - pose.bounds_min[0], pose.bounds_max[2] - pose.bounds_min[2])
    target = motion.positions + np.array([0.0, 0.5 * height, 0.0])
    track = camera.sample(
        stream(spec.seed, "camera"), ground, target, subject, frames=clip, choices=spec.camera
    )
    light = lighting.sample(
        _hdris(library, root, directory / "scene"),
        stream(spec.seed, "lighting"),
        TARGET_IRRADIANCE,
        spec.hdri,
        spec.hdri_rotation_deg,
        samples=samples,
    )
    hair = (
        None
        if entry.hair is None
        else hairstyle.sample(
            stream(spec.seed, "hairstyle"), entry.hair.regions, facial_hair=entry.hair.facial_hair
        )
    )
    return Shot(spec, directory, ground, params, surface, entry.key, pose, motion, track, clip, light, hair=hair)


def _exclusions(shot: Shot, hero_radius: float) -> list[Disk]:
    disks = [Disk(float(p[0]), float(p[2]), hero_radius) for p in shot.motion.positions[::4]]
    disks += [Disk(float(m[0, 3]), float(m[2, 3]), 0.6) for m in shot.track.samples]
    return disks


def _foreground_centre(shot: Shot, rng: np.random.Generator) -> tuple[float, float]:
    eye = shot.track.samples[len(shot.track.samples) // 2][:3, 3]
    subject = shot.motion.positions[len(shot.motion.positions) // 2]
    point = eye + (subject - eye) * rng.uniform(0.25, 0.6)
    return float(point[0]), float(point[2])


def _layers(shot: Shot, library: Library, stage: Usd.Stage, root: Path) -> list[tuple[Layer, float]]:
    rng = stream(shot.spec.seed, "scatter")
    density = float(rng.uniform(0.35, 1.6))
    density = density if shot.spec.foliage_density is None else shot.spec.foliage_density
    foreground = bool(rng.random() < 0.6)
    foreground = foreground if shot.spec.foreground_cover is None else shot.spec.foreground_cover

    def material(model: Model) -> str:
        shot.assets[model.key] = model
        return looks.model(stage, root, shot.scene_dir, model, rng)

    def asset(model: Model) -> str:
        return relative_asset(root / model.file, shot.scene_dir)

    out: list[tuple[Layer, float]] = []
    rules = [*layers.RULES, layers.FOREGROUND] if foreground else list(layers.RULES)
    for rule in rules:
        centre = _foreground_centre(shot, rng) if rule is layers.FOREGROUND else (0.0, 0.0)
        scale = density if rule.role != "rock" else 1.0
        built = layers.layer(rule, library, rng, scale, centre=centre, material=material, asset=asset)
        if built is not None:
            out.append((built, 0.0 if rule is layers.FOREGROUND else 0.9))
    return out


def _author_scatter(shot: Shot, library: Library, stage: Usd.Stage, root: Path) -> None:
    rng = stream(shot.spec.seed, "scatter", "placement")
    hero_radius = 0.5 * max(
        shot.pose.bounds_max[0] - shot.pose.bounds_min[0], shot.pose.bounds_max[2] - shot.pose.bounds_min[2]
    )
    next_id = FIRST_SCATTER_ID
    for built, keep_out in _layers(shot, library, stage, root):
        exclusions = _exclusions(shot, hero_radius * keep_out) if keep_out else _exclusions(shot, 0.0)[-1:]
        instances = scatter.place(shot.ground, built, exclusions, rng)
        if len(instances) == 0:
            continue
        grows = built.name in GROWING
        multipliers = scatter.growth(rng, len(instances), len(shot.frames) + 2) if grows else None
        grow = None if multipliers is None else (multipliers, shot.frames[0])
        scatter.author(stage, f"/World/scatter/{built.name}", built, instances, next_id, grow=grow)
        shot.layers.append(
            {
                "name": built.name,
                "first_id": next_id,
                "count": len(instances),
                "prototypes": [p.key for p in built.prototypes],
            }
        )
        next_id += len(instances)


def _terrain_layer(shot: Shot) -> str:
    path = shot.scene_dir / "terrain.usdc"
    stage = new_stage(path)
    mesh = terrain.author(stage, "/terrain", shot.ground)
    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return "./terrain.usdc"


def _author_ground(shot: Shot, library: Library, stage: Usd.Stage, root: Path) -> None:
    surface = next(s for s in library.surfaces if s.asset_id == shot.surface)
    prim = stage.DefinePrim("/World/terrain", "Mesh")
    prim.GetReferences().AddReference(_terrain_layer(shot))
    tag_instances(prim, np.array([TERRAIN_ID]))
    materials.bind(prim, looks.ground(stage, root, shot.scene_dir, surface))


def _author_hair(shot: Shot, entry: Hero, stage: Usd.Stage, root: Path, style: hairstyle.Style) -> None:
    """Grow the shot's hairstyle on the posed skin into scene/hair.usdc, under the moving hero."""
    if entry.hair is None:
        raise ValueError(f"hero {entry.key} has no hair regions")
    head = heroes.read_head(root / shot.pose.file, entry.hair.regions, entry.hair.forward)
    built = hairstyle.build(head, style, stream(shot.spec.seed, "hairstyle", "build"))
    layer = new_stage(shot.scene_dir / "hair.usdc")
    names = hairstyle.author(layer, "/hair", built)
    layer.SetDefaultPrim(layer.GetPrimAtPath("/hair"))
    layer.GetRootLayer().Save()
    stage.DefinePrim("/World/hero/hair").GetReferences().AddReference("./hair.usdc")
    scalp, facial = looks.hair(stage, style)
    for name in names:
        prim = stage.OverridePrim(f"/World/hero/hair/{name}")
        materials.bind(prim, scalp if name == "scalp" else facial)
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "instance_id", Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant
        ).Set(hero.HAIR_ID)


def write(shot: Shot, library: Library, root: Path, render: RenderParams) -> Path:
    shot.scene_dir.mkdir(parents=True, exist_ok=True)
    path = shot.scene_dir / "shot.usda"
    stage = new_stage(path)
    stage.SetStartTimeCode(shot.frames[0])
    stage.SetEndTimeCode(shot.frames[-1])
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    _author_ground(shot, library, stage, root)
    skin, groom = looks.hero(stage, root, shot.scene_dir, library.hero(shot.hero_key))
    coat = hero.sample_coat(stream(shot.spec.seed, "hero", "coat"), len(shot.frames) + 2)
    for material in (skin, groom):
        materials.animate_tint(material, coat, shot.frames[0])
    hero.author(
        stage,
        "/World/hero",
        relative_asset(root / shot.pose.file, shot.scene_dir),
        shot.motion,
        first_frame=shot.frames[0],
        skin=skin,
        groom=groom,
    )
    if shot.hair is not None:
        _author_hair(shot, library.hero(shot.hero_key), stage, root, shot.hair)
    _author_scatter(shot, library, stage, root)
    camera.author(stage, "/World/camera", shot.track)
    lighting.author(stage, "/World/lights", shot.light, shot.frames[0])
    settings.author(
        stage,
        "/World/camera",
        settings.RenderConfig(
            render.width,
            render.height,
            render.path_traced_samples,
            shot.frames,
            shot.track.motion_blur,
            shot.track.depth_of_field,
        ),
    )
    stage.GetRootLayer().Save()
    return path
