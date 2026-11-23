"""Compose one shot: terrain, hero, camera, scatter, lights and render settings into USD layers."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.fields import Ground
from strata_datagen.generators import (
    camera,
    hairstyle,
    hero,
    human,
    lighting,
    materials,
    scatter,
    terrain,
    thin,
)
from strata_datagen.generators.scatter import Disk, Layer
from strata_datagen.library.catalogue import Hero, HeroPose, Library, Model
from strata_datagen.render import settings
from strata_datagen.rng import stream
from strata_datagen.scene import creature, deformation, hair, layers, looks
from strata_datagen.scene.spec import DEFAULT_FRAMES, REFERENCE_WIDTH_PX, ShotSpec, frames
from strata_datagen.timeline import FPS
from strata_datagen.usdutil import new_stage, relative_asset, tag_instances

TERRAIN_ID = 1
FIRST_THIN_ID = 100
FIRST_SCATTER_ID = 1000
TARGET_IRRADIANCE = 3.0
GROWING = ("grass_near", "grass_far", "groundcover", "shrubs")


@dataclass(frozen=True)
class RenderParams:
    width: int
    height: int
    path_traced_samples: int


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
    thin: list[thin.Structure]
    placement: human.ClipPlacement | None
    layers: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    assets: dict[str, Model] = field(default_factory=dict[str, Model])
    hair: hairstyle.Style | None = None
    hero_animation: dict[str, object] = field(default_factory=dict[str, object])

    @property
    def scene_dir(self) -> Path:
        return self.directory / "scene"

    @property
    def groomed(self) -> bool:
        """Only a library hero's pose file holds a groom, and only when it grew hairs."""
        return self.placement is None and self.pose.hairs > 0


def samples_for(base: int, track: camera.CameraTrack) -> int:
    """Depth of field or motion blur doubles the samples, measured to match a still shot's error."""
    return base * 2 if track.depth_of_field or track.motion_blur else base


def _choice(rng: np.random.Generator, options: list[str], fixed: str | None) -> str:
    drawn = options[int(rng.integers(0, len(options)))]
    return drawn if fixed is None else fixed


def _hero_pose(
    library: Library, spec: ShotSpec, rng: np.random.Generator, frames: int
) -> tuple[Hero, HeroPose]:
    entry = library.hero(_choice(rng, sorted(h.key for h in library.heroes), spec.hero))
    fits = sorted(p.pose for p in entry.poses if p.frames == 0 or p.frames >= frames)
    if not fits:
        raise ValueError(f"hero {entry.key} has no clip of at least {frames} frames")
    pose = _choice(rng, fits, spec.pose)
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


def thin_view(track: camera.CameraTrack, motion: hero.HeroMotion, radius: float, height: float) -> thin.View:
    focal = track.lens.focal_mm
    return thin.View(
        cameras=track.samples,
        tan_half=(0.5 * camera.SENSOR_MM[0] / focal, 0.5 * camera.SENSOR_MM[1] / focal),
        pixel_rad=camera.SENSOR_MM[0] / focal / REFERENCE_WIDTH_PX,
        subject=motion.positions,
        subject_radius_m=radius,
        subject_height_m=height,
    )


def plan(spec: ShotSpec, library: Library, root: Path, frame_count: int = DEFAULT_FRAMES) -> Shot:
    directory = root / "shots" / spec.shot_id
    clip = frames(spec, frame_count)
    params = terrain.sample_params(stream(spec.seed, "terrain"))
    ground = terrain.build(params, stream(spec.seed, "terrain", "noise"))
    surface = _choice(stream(spec.seed, "ground"), sorted(s.asset_id for s in library.surfaces), spec.ground)
    entry, pose = _hero_pose(library, spec, stream(spec.seed, "hero"), len(clip))
    samples = len(clip) + 2
    placement = None
    if pose.frames:
        placement = human.place(stream(spec.seed, "hero", "clip"), ground, np.asarray(pose.track), len(clip))
        yaw = np.full(samples, placement.yaw_deg)
        motion = hero.HeroMotion(placement.positions, yaw, placement.speed_mps)
    else:
        motion = hero.sample_motion(
            stream(spec.seed, "hero", "motion"), ground, samples, -pose.bounds_min[1], spec.hero_speed_mps
        )
        walk = creature.clip_gait(root / pose.file)
        if walk is not None:
            motion = creature.ground(motion, walk, ground, stream(spec.seed, "hero", "gait"))
    height = pose.bounds_max[1] - pose.bounds_min[1]
    subject = max(high - low for low, high in zip(pose.bounds_min, pose.bounds_max, strict=True))
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
    view = thin_view(track, motion, 0.5 * subject, height)
    structures = thin.sample(stream(spec.seed, "thin"), ground, view, spec.thin)
    hair = (
        None
        if entry.hair is None
        else hairstyle.sample(
            stream(spec.seed, "hairstyle"),
            entry.hair.regions,
            facial_hair=entry.hair.facial_hair,
            duration_s=(clip[-1] - clip[0]) / FPS,
        )
    )
    return Shot(
        spec,
        directory,
        ground,
        params,
        surface,
        entry.key,
        pose,
        motion,
        track,
        clip,
        light,
        structures,
        placement,
        hair=hair,
    )


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


def _author_thin(shot: Shot, stage: Usd.Stage) -> None:
    """Each structure is its own binary layer, referenced here and bound to its finishes."""
    next_id = FIRST_THIN_ID
    for index, structure in enumerate(shot.thin):
        name = f"thin_{index}_{structure.kind}"
        layer = new_stage(shot.scene_dir / f"{name}.usdc")
        parts = thin.author(
            layer,
            "/structure",
            structure,
            next_id,
            name=name,
            first_frame=shot.frames[0],
            samples=len(shot.track.samples),
        )
        layer.SetDefaultPrim(layer.GetPrimAtPath("/structure"))
        layer.GetRootLayer().Save()
        stage.DefinePrim(f"/World/thin/{name}").GetReferences().AddReference(f"./{name}.usdc")
        finishes = {"strands": structure.strand_finish, "members": structure.member_finish}
        for part in parts:
            material = looks.thin_finish(stage, finishes[part.name])
            for prim in part.prims:
                materials.bind(
                    stage.OverridePrim(prim.replace("/structure", f"/World/thin/{name}", 1)), material
                )
            shot.layers.append({"name": f"{name}_{part.name}", "first_id": part.instance_id, "count": 1})
            next_id = part.instance_id + 1


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
    tag_instances(prim, np.array([TERRAIN_ID]), "terrain")
    materials.bind(prim, looks.ground(stage, root, shot.scene_dir, surface))


def write(shot: Shot, library: Library, root: Path, render: RenderParams) -> Path:
    shot.scene_dir.mkdir(parents=True, exist_ok=True)
    path = shot.scene_dir / "shot.usda"
    stage = new_stage(path)
    stage.SetStartTimeCode(shot.frames[0])
    stage.SetEndTimeCode(shot.frames[-1])
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    _author_ground(shot, library, stage, root)
    asset = relative_asset(root / shot.pose.file, shot.scene_dir)
    if shot.placement is None:
        skin, groom = looks.hero(stage, root, shot.scene_dir, library.hero(shot.hero_key))
        coat = hero.sample_coat(stream(shot.spec.seed, "hero", "coat"), len(shot.frames) + 2)
        for material in (skin, groom):
            materials.animate_tint(material, coat, shot.frames[0])
        hero.author(
            stage, "/World/hero", asset, shot.motion, first_frame=shot.frames[0], skin=skin, groom=groom
        )
        shot.hero_animation = creature.animate(
            stage,
            "/World/hero",
            root / shot.pose.file,
            asset_path=asset,
            frames=shot.frames,
            seed=shot.spec.seed,
        )
    else:
        human.author(
            stage,
            "/World/hero",
            asset,
            shot.placement,
            first_frame=shot.frames[0],
            instance_id=hero.SKIN_ID,
            name="hero_skin",
        )
    if shot.hair is not None:
        entry = library.hero(shot.hero_key)
        if entry.hair is None:
            raise ValueError(f"hero {entry.key} has no hair regions")
        hair.author(
            stage,
            shot.scene_dir,
            root / shot.pose.file,
            entry.hair,
            shot.hair,
            frames=shot.frames,
            placement=shot.placement,
            seed=shot.spec.seed,
        )
    _author_scatter(shot, library, stage, root)
    _author_thin(shot, stage)
    deformation.apply(stage)
    camera.author(stage, "/World/camera", shot.track, render.width, render.height)
    lighting.author(stage, "/World/lights", shot.light, shot.frames[0])
    settings.author(
        stage,
        "/World/camera",
        settings.RenderConfig(
            render.width,
            render.height,
            samples_for(render.path_traced_samples, shot.track),
            shot.frames,
            shot.track.motion_blur,
            shot.track.depth_of_field,
        ),
    )
    stage.GetRootLayer().Save()
    return path
