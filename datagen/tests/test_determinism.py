import filecmp
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.library.catalogue import Hdri, Hero, HeroPose, Library, Model, Provenance, Surface
from strata_datagen.presets import PRESETS
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import DEFAULT_FRAMES, ShotSpec, frames
from strata_datagen.usdutil import ints, new_stage, vec3f
from strata_dataset import roles

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


def hero_files(root: Path, lib: Library) -> None:
    """Static hero pose files with a one-triangle skin, so the scene can inspect them."""
    for pose in lib.heroes[0].poses:
        path = root / pose.file
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = new_stage(path)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/hero").GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/hero/skin")
        mesh.CreatePointsAttr(vec3f(np.eye(3)))
        mesh.CreateFaceVertexCountsAttr(ints(np.array([3])))
        mesh.CreateFaceVertexIndicesAttr(ints(np.arange(3)))
        stage.GetRootLayer().Save()


def build(root: Path, spec: ShotSpec) -> Path:
    lib = library()
    hero_files(root, lib)
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

    def test_a_groomed_hero_declares_its_groom_id(self) -> None:
        with TemporaryDirectory() as tmp:
            record = json.loads((build(Path(tmp), ShotSpec(seed=7)) / "manifest.json").read_text())
            self.assertIn({"name": "hero_groom", "first_id": 3, "count": 1}, record["instance_ids"])

    def test_every_tagged_surface_carries_the_roles_of_its_manifest_id_range(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = build(Path(tmp), ShotSpec(seed=7))
            record = json.loads((directory / "manifest.json").read_text())
            names = {
                first + k: r["name"]
                for r in record["instance_ids"]
                for first in [r["first_id"]]
                for k in range(r["count"])
            }
            stage = Usd.Stage.Open(str(directory / "scene" / "shot.usda"))
            tagged = 0
            for prim in stage.Traverse():
                api = UsdGeom.PrimvarsAPI(prim)
                if not api.GetPrimvar("instance_id").HasAuthoredValue():
                    continue
                ids = np.atleast_1d(np.asarray(api.GetPrimvar("instance_id").Get()))
                found = np.stack([np.atleast_1d(np.asarray(api.GetPrimvar(r).Get())) for r in roles.ROLES], 1)
                expected = np.array([roles.membership(names[int(i)]) for i in ids])
                np.testing.assert_array_equal(
                    np.broadcast_to(found, expected.shape), expected, str(prim.GetPath())
                )
                tagged += 1
            self.assertGreater(tagged, 5)
            self.assertEqual({names[i] for i in (0, 1, 2, 3)}, {"sky", "terrain", "hero_skin", "hero_groom"})

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
                hero_files(Path(tmp), lib)
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
