from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np

from strata_training.augment.camera import AP1_LUMINANCE, CODECS, CameraParams, degrade
from strata_training.augment.deep import Composition, DeepSamples, Layer, Shot, compose
from strata_training.augment.grain import ISO_STOPS, PROFILES, GrainParams, apply_grain
from strata_training.augment.occluders import (
    KINDS,
    Occluder,
    random_colour,
    random_shape,
    rasterize,
    samples,
    tilted_depth,
)
from strata_training.types import Array, Layers

Range = tuple[float, float]


@dataclass(frozen=True)
class AugmentConfig:
    clean: float = 0.2
    recompose: float = 0.35
    hero: float = 0.5
    scatter_count: tuple[int, int] = (1, 8)
    element_front: float = 0.5
    occluders: float = 0.35
    occluder_count: tuple[int, int] = (1, 3)
    occluder_front: float = 0.5
    occluder_motion: float = 0.3
    occluder_defocus: float = 0.3
    occluder_textured: float = 0.5
    instance_grade: float = 0.3
    instance_count: tuple[int, int] = (1, 3)
    grade_stops: float = 0.5
    grade_saturation: Range = (0.6, 1.4)
    scene_grade: float = 0.5
    exposure_stops: Range = (-1.0, 1.0)
    white_balance_stops: float = 0.15
    grain: float = 0.5
    grain_profiles: tuple[str, ...] = tuple(PROFILES)
    grain_isos: tuple[int, ...] = (400, 640, 800, 1280, 1600, 2560, 3200)
    grain_exposure: Range = (-2.0, 2.0)
    grain_amount: Range = (0.5, 1.5)
    camera: float = 0.7
    contrast: Range = (0.85, 1.15)
    saturation: Range = (0.8, 1.2)
    logc3: float = 0.5
    white: Range = (4.0, 16.0)
    sharpen: float = 0.5
    sharpen_amount: Range = (0.1, 0.8)
    sharpen_sigma: Range = (0.6, 1.5)
    codec_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15)
    jpeg_quality: tuple[int, int] = (30, 95)
    crf: tuple[int, int] = (18, 38)
    ten_bit: float = 0.3
    layers: int = 4

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AugmentConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown augmentation settings: {sorted(unknown)}")
        converted: dict[str, Any] = {
            k: tuple(cast(list[Any], v)) if isinstance(v, list) else v for k, v in values.items()
        }
        config = cls(**converted)
        if not set(config.grain_profiles) <= set(PROFILES) or not set(config.grain_isos) <= set(
            ISO_STOPS
        ):
            raise ValueError("Grain profiles and ISOs must come from the studio group")
        if len(config.codec_weights) != len(CODECS) or min(config.codec_weights) < 0:
            raise ValueError("codec_weights needs one nonnegative weight per codec")
        return config


@dataclass(frozen=True)
class Roles:
    hero: frozenset[int]
    scatter: tuple[tuple[int, int], ...]

    def is_scatter(self, ids: Array) -> Array:
        found = np.zeros(ids.shape, dtype=bool)
        for first, count in self.scatter:
            found |= (ids >= first) & (ids < first + count)
        return found


def roles(manifest: dict[str, Any]) -> Roles:
    entries: list[dict[str, Any]] = manifest["instance_ids"]
    hero = frozenset(
        int(e["first_id"]) + i
        for e in entries
        if e["name"].startswith("hero")
        for i in range(e["count"])
    )
    scatter = tuple((int(e["first_id"]), int(e["count"])) for e in entries if "prototypes" in e)
    return Roles(hero, scatter)


@dataclass(frozen=True)
class Donor:
    shot: Shot
    roles: Roles


@dataclass(frozen=True)
class AugmentedSample:
    image: Array
    clean: Array
    layers: Layers
    record: dict[str, Any]


def augment_sample(
    shot: Shot,
    shot_roles: Roles,
    donor: Donor | None,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> AugmentedSample:
    record: dict[str, Any] = {}
    if rng.random() < config.clean:
        record["clean"] = True
        layers = shot.deep.targets(config.layers)
        return AugmentedSample(shot.denoised, shot.denoised, layers, record)
    composition = recompose(shot, shot_roles, donor, rng, config, record)
    clean = composition.image
    image = photometric(clean, rng, config, record)
    return AugmentedSample(image, clean, composition.samples.targets(config.layers), record)


def recompose(
    shot: Shot,
    shot_roles: Roles,
    donor: Donor | None,
    rng: np.random.Generator,
    config: AugmentConfig,
    record: dict[str, Any],
) -> Composition:
    scene = np.eye(3)
    if rng.random() < config.scene_grade:
        stops = rng.uniform(*config.exposure_stops)
        balance = rng.normal(0, config.white_balance_stops, 2)
        gains = 2.0 ** np.array([stops + balance[0], stops, stops + balance[1]])
        scene = np.diag(gains)
        record["scene_gain"] = gains.tolist()
    layers = _graded_background(shot, shot_roles, rng, config, record)
    if donor is not None and rng.random() < config.recompose:
        if donor.shot.deep.shape != shot.deep.shape:
            raise ValueError("Recomposition donors must share the pixel grid")
        element = _element(shot, donor, rng, config, record)
        if element is not None:
            layers.append(element)
    if rng.random() < config.occluders:
        layers.append(Layer(_occluders(shot, rng, config, record), None))
    scoped = [Layer(layer.samples, layer.shot, scene @ layer.matrix) for layer in layers]
    return compose(scoped, shot, scene)


def _grade(rng: np.random.Generator, config: AugmentConfig) -> Array:
    gains = 2.0 ** rng.normal(0, config.grade_stops, 3)
    saturation = rng.uniform(*config.grade_saturation)
    luminance = np.outer(np.ones(3), AP1_LUMINANCE)
    return np.diag(gains) @ (luminance + saturation * (np.eye(3) - luminance))


def _graded_background(
    shot: Shot,
    shot_roles: Roles,
    rng: np.random.Generator,
    config: AugmentConfig,
    record: dict[str, Any],
) -> list[Layer]:
    deep = shot.deep
    present = np.unique(deep.ids)
    if rng.random() >= config.instance_grade or not present.size:
        return [Layer(deep, shot)]
    count = min(
        int(rng.integers(config.instance_count[0], config.instance_count[1] + 1)), present.size
    )
    chosen = rng.choice(present, count, replace=False)
    layers: list[Layer] = []
    remaining = np.ones(deep.z.size, dtype=bool)
    for instance in chosen:
        mask = deep.ids == instance
        if instance in shot_roles.hero:
            mask = np.isin(deep.ids, list(shot_roles.hero))
        mask &= remaining
        remaining &= ~mask
        layers.append(Layer(deep.select(mask), shot, _grade(rng, config)))
    layers.append(Layer(deep.select(remaining), shot))
    record["graded_instances"] = [int(i) for i in chosen]
    return layers


def _element(
    shot: Shot,
    donor: Donor,
    rng: np.random.Generator,
    config: AugmentConfig,
    record: dict[str, Any],
) -> Layer | None:
    deep = donor.shot.deep
    hero = np.isin(deep.ids, list(donor.roles.hero))
    scatter_ids = np.unique(deep.ids[donor.roles.is_scatter(deep.ids)])
    if hero.any() and (rng.random() < config.hero or not scatter_ids.size):
        chosen = hero
        record["element"] = "hero"
    elif scatter_ids.size:
        count = min(
            int(rng.integers(config.scatter_count[0], config.scatter_count[1] + 1)),
            scatter_ids.size,
        )
        picked = rng.choice(scatter_ids, count, replace=False)
        chosen = np.isin(deep.ids, picked)
        record["element"] = [int(i) for i in picked]
    else:
        return None
    keep = chosen | _in_front_of(deep, chosen)
    element = deep.select(keep)
    scale = _placement(shot.deep, element, rng.random() < config.element_front, rng)
    if scale is None:
        return None
    record["element_depth_scale"] = scale
    return Layer(element.scaled(scale), donor.shot, _maybe_grade(rng, config, record))


def _maybe_grade(rng: np.random.Generator, config: AugmentConfig, record: dict[str, Any]) -> Array:
    if rng.random() < config.instance_grade:
        record["element_graded"] = True
        return _grade(rng, config)
    return np.eye(3)


def _in_front_of(deep: DeepSamples, chosen: Array) -> Array:
    pixels = deep.shape[0] * deep.shape[1]
    deepest = np.full(pixels, -np.inf)
    np.maximum.at(deepest, deep.pixel[chosen], deep.z[chosen])
    return deep.z < deepest[deep.pixel]


def _placement(
    background: DeepSamples, element: DeepSamples, front: bool, rng: np.random.Generator
) -> float | None:
    both = (background.counts > 0) & (element.counts > 0)
    if not both.any():
        return None
    near = background.z[background.offsets[:-1][both]]
    far = background.z[background.offsets[1:][both] - 1]
    element_near = element.z[element.offsets[:-1][both]]
    element_far = element.z[element.offsets[1:][both] - 1]
    in_front = float(np.min(near / element_far))
    if front:
        return in_front * float(rng.uniform(0.6, 0.95))
    spread = max(float(np.median(far / element_near)), in_front * 1.01)
    return float(np.exp(rng.uniform(np.log(in_front), np.log(spread))))


def _occluders(
    shot: Shot, rng: np.random.Generator, config: AugmentConfig, record: dict[str, Any]
) -> DeepSamples:
    deep = shot.deep
    size = deep.shape
    filled = deep.counts > 0
    nearest = deep.z[deep.offsets[:-1][filled]]
    farthest = deep.z[deep.offsets[1:][filled] - 1]
    near, far = (float(np.quantile(nearest, 0.01)), float(np.quantile(farthest, 0.95)))
    luminance = float(np.tensordot(AP1_LUMINANCE, shot.denoised, axes=1).mean())
    made: list[Occluder] = []
    kinds: list[str] = []
    for _ in range(int(rng.integers(config.occluder_count[0], config.occluder_count[1] + 1))):
        kind = str(rng.choice(KINDS))
        motion = (0.0, 0.0)
        if rng.random() < config.occluder_motion:
            motion = (float(rng.normal(0, 8)), float(rng.normal(0, 8)))
        defocus = float(rng.uniform(0.5, 6)) if rng.random() < config.occluder_defocus else 0.0
        coverage = rasterize(random_shape(kind, size, rng), size, motion, defocus)
        textured = bool(rng.random() < config.occluder_textured)
        colour = random_colour(size, luminance, textured, rng)
        if rng.random() < config.occluder_front:
            depth = near * float(rng.uniform(0.3, 0.9))
        else:
            depth = float(np.exp(rng.uniform(np.log(near), np.log(max(far, near * 1.01)))))
        made.append(Occluder(coverage, colour, tilted_depth(size, depth, rng)))
        kinds.append(kind)
    record["occluders"] = kinds
    return samples(made, size)


def photometric(
    image: Array, rng: np.random.Generator, config: AugmentConfig, record: dict[str, Any]
) -> Array:
    result = image
    if rng.random() < config.grain:
        params = GrainParams(
            str(rng.choice(config.grain_profiles)),
            int(rng.choice(config.grain_isos)),
            float(rng.uniform(*config.grain_exposure)),
            int(rng.integers(0, 2048)),
            float(rng.uniform(*config.grain_amount)),
        )
        result = apply_grain(result, params)
        record["grain"] = params.__dict__
    if rng.random() < config.camera:
        weights = np.array(config.codec_weights) / sum(config.codec_weights)
        codec = str(rng.choice(CODECS, p=weights))
        if codec == "jpeg":
            quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
        else:
            quality = int(rng.integers(config.crf[0], config.crf[1] + 1))
        bits = 10 if codec in ("none", "h265") and rng.random() < config.ten_bit else 8
        sharpen = rng.random() < config.sharpen
        params = CameraParams(
            float(rng.uniform(*config.contrast)),
            float(rng.uniform(*config.saturation)),
            "logc3" if rng.random() < config.logc3 else "rec709",
            float(rng.uniform(*config.white)),
            float(rng.uniform(*config.sharpen_amount)) if sharpen else 0.0,
            float(rng.uniform(*config.sharpen_sigma)),
            codec,
            quality,
            bits,
        )
        result = degrade(result, params)
        record["camera"] = params.__dict__
    return result.astype(np.float32)
