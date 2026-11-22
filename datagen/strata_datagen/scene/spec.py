"""What a shot is made of before sampling: a seed plus optional fixed choices (presets set these)."""

from dataclasses import dataclass, field

from strata_datagen.generators.camera import CameraChoices

RECIPE_VERSION = "strata-datagen-recipe-0.3.0"
DEFAULT_FRAMES = 720


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


def frames(spec: ShotSpec, default: int = DEFAULT_FRAMES) -> tuple[int, ...]:
    """Frame numbers of a shot: long continuous shots by default, presets may fix a length."""
    return tuple(range(1, (default if spec.frame_count is None else spec.frame_count) + 1))
