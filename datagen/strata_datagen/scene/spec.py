"""What a shot is made of before sampling: a seed plus optional fixed choices (presets set these)."""

from dataclasses import dataclass, field

import numpy as np

from strata_datagen.generators.camera import CameraChoices

RECIPE_VERSION = "strata-datagen-recipe-0.2.0"
SHORT_FRAMES = 8
LONG_FRAMES = (24, 96)
LONG_SHARE = 0.1


@dataclass(frozen=True)
class ShotSpec:
    seed: int
    name: str = "shot"
    hero: str | None = None
    pose: str | None = None
    hero_speed_mps: float | None = None
    ground: str | None = None
    hdri: str | None = None
    hdri_rotation_deg: float | None = None
    foliage_density: float | None = None
    foreground_cover: bool | None = None
    frame_count: int | None = None
    camera: CameraChoices = field(default_factory=CameraChoices)

    @property
    def shot_id(self) -> str:
        return f"{self.name}_{self.seed:06d}"


def frames(spec: ShotSpec, long_share: float, rng: np.random.Generator) -> tuple[int, ...]:
    """Short clips by default; a long_share of shots are long sequences for temporal training."""
    long = bool(rng.random() < long_share)
    count = int(rng.integers(LONG_FRAMES[0], LONG_FRAMES[1] + 1)) if long else SHORT_FRAMES
    return tuple(range(1, (count if spec.frame_count is None else spec.frame_count) + 1))
