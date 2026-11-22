import filecmp
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from strata_datagen.library.catalogue import Hdri, Hero, HeroPose, Library, Model, Provenance, Surface
from strata_datagen.presets import PRESETS
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import DEFAULT_FRAMES, ShotSpec, frames

PROVENANCE = Provenance("test", "test", "test")
TEXTURES = {
    "albedo": "assets/t/albedo.jpg",
    "roughness": "assets/t/roughness.png",
    "normal": "assets/t/normal.png",
    "opacity": "assets/t/opacity.png",
}
RENDER = compose.RenderParams(320, 180, 8)


def _model(key: str, role: str) -> Model:
    return Model(
        key,
        key.split("_", maxsplit=1)[0],
        key,
        role,
        f"assets/m/{key}.usdc",
        (0.6, 0.5, 0.6),
        100,
        TEXTURES,
        PROVENANCE,
    )


def library() -> Library:
    roles = ("grass", "groundcover", "shrub", "rock", "debris")
    models = tuple(_model(f"{role}{i}_v1", role) for role in roles for i in range(3))
    surfaces = (Surface("soil", "soil", 2.0, TEXTURES, PROVENANCE),)
    hdris = tuple(
        Hdri(k, f"assets/h/{k}.exr", (0.3, 0.6, 0.74), 5.0, (1.0, 0.9, 0.8), 1.0, PROVENANCE)
        for k in ("sky", "woods")
    )
    poses = tuple(
        HeroPose(p, f"assets/hero/{p}.usdc", (-0.8, -0.1, -0.3), (0.8, 0.7, 0.4), 10) for p in ("a", "b")
    )
    return Library(models, surfaces, hdris, (Hero("creature", "assets/hero/albedo.jpg", poses, PROVENANCE),))


def build(root: Path, spec: ShotSpec) -> Path:
    lib = library()
    shot = compose.plan(spec, lib, root)
    compose.write(shot, lib, root, RENDER)
    export.write(shot, lib, RENDER)
    return shot.directory


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


class DeterminismTest(unittest.TestCase):
    def test_same_seed_gives_identical_files(self) -> None:
        with TemporaryDirectory() as tmp:
            first = build(Path(tmp) / "a", ShotSpec(seed=7))
            second = build(Path(tmp) / "b", ShotSpec(seed=7))
            names = _files(first)
            self.assertEqual(names, _files(second))
            self.assertIn("scene/shot.usda", names)
            self.assertIn("scene/terrain.usdc", names)
            _, mismatch, errors = filecmp.cmpfiles(first, second, names, shallow=False)
            self.assertEqual((mismatch, errors), ([], []))

    def test_different_seeds_differ(self) -> None:
        with TemporaryDirectory() as tmp:
            first = build(Path(tmp), ShotSpec(seed=1))
            second = build(Path(tmp), ShotSpec(seed=2))
            usd = "scene/shot.usda"
            self.assertNotEqual((first / usd).read_bytes(), (second / usd).read_bytes())

    def test_presets_fix_their_choices(self) -> None:
        lib = library()
        for spec in PRESETS.values():
            fixed = ShotSpec(
                seed=spec.seed,
                name=spec.name,
                hero="creature",
                pose="a",
                hero_speed_mps=spec.hero_speed_mps,
                ground="soil",
                hdri="woods",
                hdri_rotation_deg=spec.hdri_rotation_deg,
                foliage_density=spec.foliage_density,
                foreground_cover=spec.foreground_cover,
                camera=spec.camera,
            )
            with TemporaryDirectory() as tmp:
                shot = compose.plan(fixed, lib, Path(tmp))
            self.assertEqual(shot.track.move, spec.camera.move)
            self.assertEqual(shot.light.hdri.key, "woods")
            self.assertEqual(shot.track.motion_blur, spec.camera.motion_blur)


if __name__ == "__main__":
    unittest.main()


class ShotLengthTest(unittest.TestCase):
    def test_shots_are_long_by_default_and_presets_fix_their_length(self) -> None:
        self.assertEqual(len(frames(ShotSpec(seed=1))), DEFAULT_FRAMES)
        self.assertEqual(len(frames(ShotSpec(seed=1), 100)), 100)
        self.assertEqual(len(frames(PRESETS["foliage_heavy"])), 24)
