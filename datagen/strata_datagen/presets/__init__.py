"""Fixed recipes: canary shots and generation presets that always build the same scene.

Assets are named by their external-library ids; everything not fixed here comes from the seed.
"""

from strata_datagen.generators.camera import CameraChoices
from strata_datagen.scene.spec import ShotSpec

PRESETS: dict[str, ShotSpec] = {
    "hero_in_terrain": ShotSpec(
        seed=101,
        name="hero_in_terrain",
        hero="jaguar",
        pose="neutral",
        hero_speed_mps=0.0,
        ground="ompjC0",
        hdri="syferfontein_6d_clear_puresky",
        hdri_rotation_deg=120.0,
        foliage_density=0.7,
        foreground_cover=False,
        frame_count=24,
        camera=CameraChoices(
            move="orbit",
            distance_m=4.5,
            height_m=0.9,
            azimuth_deg=40.0,
            fill=0.6,
            depth_of_field=False,
            motion_blur=False,
        ),
    ),
    "foliage_heavy": ShotSpec(
        seed=202,
        name="foliage_heavy",
        hero="jaguar",
        pose="look_left",
        hero_speed_mps=0.5,
        ground="vcpoajjs",
        hdri="woods",
        hdri_rotation_deg=200.0,
        foliage_density=1.6,
        foreground_cover=True,
        frame_count=24,
        camera=CameraChoices(
            move="dolly",
            distance_m=5.0,
            height_m=0.45,
            azimuth_deg=-60.0,
            fill=0.5,
            depth_of_field=True,
            motion_blur=True,
        ),
    ),
}
