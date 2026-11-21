"""What a shot is made of before sampling: a seed plus optional fixed choices (presets set these)."""

from dataclasses import dataclass, field

from strata_datagen.generators.camera import CameraChoices

RECIPE_VERSION = "strata-datagen-recipe-0.1.0"
FRAMES = tuple(range(1, 25))


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
    camera: CameraChoices = field(default_factory=CameraChoices)

    @property
    def shot_id(self) -> str:
        return f"{self.name}_{self.seed:06d}"
