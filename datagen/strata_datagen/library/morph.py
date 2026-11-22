"""Morphing humans: long continuous takes whose body drifts smoothly between corner humans.

A family is a few corner humans built by MPFB with one shared outfit and eyes, so every part has
the same topology in every corner. A take blends the corners' rest points, rigs and skin textures
with seeded smooth weights (curves.mixture) while one chained CMU motion (motion_chain) plays, and
bakes the result to points: gender, age, body shape, height, skin tone and face all change
gradually, clothes follow the body because they are skinned and blended the same way, and
feet stay on the ground because root travel scales with the blended hip height.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from pxr import Usd

from strata_datagen import files
from strata_datagen.generators import materials
from strata_datagen.library import curves, human_usd, humans, mocap, motion_chain, retarget
from strata_datagen.rng import stream
from strata_datagen.usdutil import FloatArray

GROUND_WINDOW_S = 1.0
PHENOTYPE = ("gender", "age", "muscle", "weight", "proportions", "height")


@dataclass(frozen=True)
class MorphRecipe:
    """Recipe block `morph`: families of `corners` humans, `takes` per family of `seconds` each."""

    first_index: int
    families: int
    corners: int
    takes: int
    seconds: float
    period_s: tuple[float, float]
    sharpness: float

    @staticmethod
    def parse(raw: dict[str, object]) -> "MorphRecipe":
        low, high = cast("list[float]", raw["period_s"])
        return MorphRecipe(
            first_index=int(cast("int", raw["first_index"])),
            families=int(cast("int", raw["families"])),
            corners=int(cast("int", raw["corners"])),
            takes=int(cast("int", raw["takes"])),
            seconds=float(cast("float", raw["seconds"])),
            period_s=(float(low), float(high)),
            sharpness=float(cast("float", raw["sharpness"])),
        )


def corner_specs(
    recipe: humans.Recipe, morph: MorphRecipe, family: int, catalog: humans.Catalog
) -> list[humans.HumanSpec]:
    """Corners of a family: independent humans dressed in the first corner's outfit and eyes."""
    first = morph.first_index + family * morph.corners
    specs = [humans.sample(recipe, first + k, catalog) for k in range(morph.corners)]
    key = f"{recipe.key_prefix}morph{family:03d}"
    return [
        replace(s, key=f"{key}_c{k}", outfit=specs[0].outfit, eyes=specs[0].eyes, garments=())
        for k, s in enumerate(specs)
    ]


def _ground(lowest: FloatArray, samples_per_second: float) -> FloatArray:
    """Height to subtract per sample: the moving minimum of the lowest point, smoothed."""
    half = int(GROUND_WINDOW_S * samples_per_second / 2)
    padded = np.pad(lowest, half, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).min(axis=1)
    kernel = np.hanning(2 * half + 3)[1:-1]
    smooth = np.convolve(np.pad(window, half, mode="edge"), kernel / kernel.sum(), mode="valid")
    return smooth[: len(lowest)]


@dataclass(frozen=True)
class Take:
    name: str
    times: FloatArray
    samples: dict[str, FloatArray]
    weights: FloatArray
    track: FloatArray
    bounds_min: FloatArray
    bounds_max: FloatArray
    windows: list[motion_chain.Window]


def bake(
    corners: list[humans.RestHuman],
    rng: np.random.Generator,
    morph: MorphRecipe,
    cmu: Path,
    *,
    takes: dict[str, list[tuple[str, int]]],
    name: str,
) -> Take:
    frames = round(morph.seconds * humans.FPS)
    times = (np.arange(frames + 2, dtype=np.float64)[:, None] + humans.SUBFRAMES[None, :]).ravel()
    lengths = {trial: _length(cmu, trial) for options in takes.values() for trial, _ in options}
    windows = motion_chain.plan(rng, takes, lengths, (frames + 3) / humans.FPS)
    skeleton, motion = motion_chain.chain(cmu, windows)
    step = mocap.FPS / humans.FPS
    source = (times - times[0]) * step
    weights = curves.mixture(rng, times / humans.FPS, len(corners), morph.period_s, morph.sharpness)
    rig_heads = np.stack([c.rig.heads for c in corners])
    mean_rig = replace(
        corners[0].rig,
        heads=rig_heads.mean(axis=0),
        tails=np.stack([c.rig.tails for c in corners]).mean(axis=0),
    )
    world, root = retarget.world_rotations(mean_rig, skeleton, motion, source)
    heads = np.einsum("sk,kbd->sbd", weights, rig_heads)
    pose = retarget.Pose(
        world,
        retarget.joint_heads(mean_rig.parents, heads, world, root * retarget.hip_scale(heads, skeleton)),
    )
    samples: dict[str, FloatArray] = {}
    for index, part in enumerate(corners[0].parts):
        rest = np.einsum("sk,knd->snd", weights, np.stack([c.parts[index].points for c in corners]))
        samples[part.name] = retarget.skin(rest, part.joints, part.weights, heads, pose)
    lowest = np.min([s[:, :, 1].min(axis=1) for s in samples.values()], axis=0)
    on_frame = np.flatnonzero(np.isclose(times, np.round(times)))
    track = pose.heads[on_frame, 0].copy()
    centre = track[len(track) // 2]
    drop = _ground(lowest, humans.FPS * len(humans.SUBFRAMES))
    shift = np.stack([np.full(len(drop), centre[0]), drop, np.full(len(drop), centre[2])], axis=-1)
    samples = {part: points - shift[:, None] for part, points in samples.items()}
    track = track - np.array([centre[0], 0.0, centre[2]])
    track[:, 1] -= drop[on_frame]
    local = [s[on_frame] - (track * np.array([1.0, 0.0, 1.0]))[:, None] for s in samples.values()]
    low: FloatArray = np.stack([s.min(axis=(0, 1)) for s in local]).min(axis=0)
    high: FloatArray = np.stack([s.max(axis=(0, 1)) for s in local]).max(axis=0)
    low[1] = 0.0
    return Take(name, times, samples, weights, track, low, high, windows)


def _length(cmu: Path, trial: str) -> int:
    return sum(1 for line in (cmu / f"{trial}.amc").read_text().splitlines() if line.strip().isdigit())


def mix_weights(weights: FloatArray) -> FloatArray:
    """Sequential mix amounts so a chain of mixes over layers 1..K-1 reproduces convex weights."""
    cumulative = np.cumsum(weights, axis=1)
    return weights[:, 1:] / np.maximum(cumulative[:, 1:], 1e-9)


def write_take(
    path: Path,
    take: Take,
    *,
    parts: list[human_usd.MeshPart],
    looks: dict[str, materials.Look],
    regions: dict[str, FloatArray],
) -> None:
    human_usd.write(path, parts=parts, samples=take.samples, times=take.times, looks=looks, regions=regions)
    stage = Usd.Stage.Open(str(path))
    mixes = mix_weights(take.weights)
    for index in range(mixes.shape[1]):
        attr = stage.GetPrimAtPath(f"{human_usd.LOOKS}/body/albedo_mix{index}").GetAttribute("inputs:mix")
        for time, value in zip(take.times, mixes[:, index], strict=True):
            attr.Set(float(value), Usd.TimeCode(float(time)))
    stage.GetRootLayer().Save()


def build_family(
    recipe: humans.Recipe, morph: MorphRecipe, family: int, library: Path, catalog: humans.Catalog
) -> Path:
    specs = corner_specs(recipe, morph, family, catalog)
    key = specs[0].key.rsplit("_c", 1)[0]
    folder = library / key
    build = folder / "build"
    build.mkdir(parents=True, exist_ok=True)
    pending = [(k, s) for k, s in enumerate(specs) if not (build / f"corner{k}.npz").exists()]
    if pending:
        humans.run_blender(
            recipe, [humans.blender_job(recipe, s, build / f"corner{k}.npz") for k, s in pending], build
        )
    corners = [humans.load_rest(build / f"corner{k}.npz") for k in range(len(specs))]
    for corner in corners[1:]:
        if [p.points.shape for p in corner.parts] != [p.points.shape for p in corners[0].parts]:
            raise ValueError(f"{key}: corners differ in topology")
    usd = folder / "usd"
    looks = {
        p.name: humans.look(
            p.kind, folder / "textures" / p.name, humans.material_file(p, specs[0].eyes, recipe), usd
        )
        for p in corners[0].parts
    }
    skins = [
        humans.look(
            "body",
            folder / "textures" / f"body_c{k}",
            humans.material_file(c.parts[0], specs[k].eyes, recipe),
            usd,
        )
        for k, c in enumerate(corners)
    ]
    textures = [s.textures for s in skins]
    if None in textures:
        raise ValueError("morph corner skins need albedo textures")
    first, *others = cast("list[materials.Textures]", textures)
    looks["body"] = replace(skins[0], textures=replace(first, albedo_layers=tuple(t.albedo for t in others)))
    regions = humans.body_regions(corners[0])
    parts = humans.mesh_parts(corners[0])
    rest = {p.name: p.points[None] for p in corners[0].parts}
    human_usd.write(usd / "rest.usdc", parts=parts, samples=rest, times=None, looks=looks, regions=regions)
    takes = {c: [(o.trial, o.start) for o in options] for c, options in recipe.motions.items()}
    clips: dict[str, object] = {}
    for index in range(morph.takes):
        name = f"morph_take{index}"
        take = bake(
            corners, stream(recipe.seed, "morph", key, str(index)), morph, recipe.cmu, takes=takes, name=name
        )
        file = usd / f"{name}.usdc"
        write_take(file, take, parts=parts, looks=looks, regions=regions)
        on_frame = np.flatnonzero(np.isclose(take.times, np.round(take.times)))
        params = {
            "weights": np.round(take.weights[on_frame], 5).tolist(),
            "phenotype": {
                p: np.round(
                    take.weights[on_frame] @ np.array([float(cast("float", s.phenotype[p])) for s in specs]),
                    5,
                ).tolist()
                for p in PHENOTYPE
            },
            "windows": [[w.trial, w.start, w.frames] for w in take.windows],
        }
        files.write_text(usd / f"{name}.params.json", json.dumps(params))
        clips[name] = {
            "file": file.relative_to(folder).as_posix(),
            "category": "morph",
            "source": "CMU mocap chain: "
            + ", ".join(f"{w.trial}@{w.start}+{w.frames}" for w in take.windows),
            "frames": len(take.track) - 2 * humans.HANDLES,
            "first_frame": humans.HANDLES,
            "fps": humans.FPS,
            "subframes": [float(v) for v in humans.SUBFRAMES],
            "speed_mps": round(
                float(np.linalg.norm(np.diff(take.track[:, [0, 2]], axis=0), axis=1).mean() * humans.FPS), 4
            ),
            "bounds_min": [round(float(v), 5) for v in take.bounds_min],
            "bounds_max": [round(float(v), 5) for v in take.bounds_max],
            "track": [[round(float(v), 5) for v in p] for p in take.track],
            "parameters": f"usd/{name}.params.json",
        }
    record = humans.manifest(specs[0], corners[0], clips, recipe.licenses)
    record.update(
        key=key,
        kind="human_morph",
        facial_hair=humans.facial_hair(specs),
        corners=[
            {"skin": s.skin.id, "phenotype": s.phenotype, "face_targets": [[t, v] for t, v in s.targets]}
            for s in specs
        ],
    )
    files.write_text(folder / "hero.json", json.dumps(record, indent=1, sort_keys=True) + "\n")
    return folder
