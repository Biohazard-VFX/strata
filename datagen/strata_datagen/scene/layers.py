"""Scatter layers for a shot: which library models grow where, and how densely."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from strata_datagen.generators.scatter import Layer, Prototype, fits
from strata_datagen.library.catalogue import Library, Model


@dataclass(frozen=True)
class LayerRule:
    name: str
    role: str
    assets: int
    density_per_m2: float
    radius_m: float
    masks: tuple[str, ...]
    footprint_range_m: tuple[float, float]
    align_to_ground: float
    sink_fraction: float


RULES = (
    LayerRule("grass_near", "grass", 3, 9.0, 14.0, ("flat", "cover"), (0.25, 0.7), 0.6, 0.05),
    LayerRule("grass_far", "grass", 2, 1.5, 60.0, ("flat", "cover"), (0.3, 0.9), 0.6, 0.05),
    LayerRule("groundcover", "groundcover", 3, 1.2, 30.0, ("flat",), (0.15, 0.5), 0.8, 0.05),
    LayerRule("shrubs", "shrub", 2, 0.06, 70.0, ("cover",), (0.6, 2.0), 0.3, 0.05),
    LayerRule("rocks", "rock", 3, 0.03, 80.0, ("low",), (0.25, 2.5), 1.0, 0.2),
    LayerRule("debris", "debris", 2, 0.02, 45.0, ("flat",), (0.4, 1.8), 1.0, 0.1),
)
FOREGROUND = LayerRule("foreground", "grass", 2, 30.0, 1.0, (), (0.25, 0.6), 0.6, 0.05)

Material = Callable[[Model], str]
Asset = Callable[[Model], str]


def _footprint(model: Model) -> float:
    return max(model.size_m[0], model.size_m[2], 1e-3)


def _prototypes(models: list[Model], material: Material, asset: Asset) -> tuple[Prototype, ...]:
    return tuple(
        Prototype(m.key, asset(m), material(m), _footprint(m), max(m.size_m[1], 1e-3)) for m in models
    )


def _choose(library: Library, rule: LayerRule, rng: np.random.Generator) -> list[Model]:
    """Pick up to rule.assets library assets, keeping the variations that fit the footprint range."""
    candidates = [m for m in library.role(rule.role) if fits(_footprint(m), rule.footprint_range_m)]
    ids = sorted({m.asset_id for m in candidates})
    if not ids:
        return []
    picked = set(rng.choice(ids, min(rule.assets, len(ids)), replace=False).tolist())
    return [m for m in candidates if m.asset_id in picked]


def layer(
    rule: LayerRule,
    library: Library,
    rng: np.random.Generator,
    density_scale: float,
    *,
    centre: tuple[float, float],
    material: Material,
    asset: Asset,
) -> Layer | None:
    models = _choose(library, rule, rng)
    if not models:
        return None
    return Layer(
        rule.name,
        _prototypes(models, material, asset),
        rule.density_per_m2 * density_scale,
        rule.radius_m,
        centre,
        rule.masks,
        rule.footprint_range_m,
        rule.align_to_ground,
        rule.sink_fraction,
    )
