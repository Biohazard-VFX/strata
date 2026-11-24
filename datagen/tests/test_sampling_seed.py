import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from jsonschema import Draft202012Validator
from pxr import Sdf, Usd

import validate
from strata_datagen import cli
from strata_datagen.generators.camera import CameraChoices
from strata_datagen.qa import checks
from strata_datagen.render import pack, settings
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import ShotSpec
from strata_dataset import records, sampling
from tests.fixtures import SKY, Sample, write_shot
from tests.test_determinism import hero_files, library


class SamplingSeedTest(unittest.TestCase):
    def test_schedule_is_stable_balanced_and_independent_of_task_boundaries(self) -> None:
        expected: tuple[tuple[int, sampling.Schedule], ...] = (
            (0, {"mode": "fixed", "seeds": [1427725051] * 3}),
            (2, {"mode": "per-frame", "seeds": [727542660, 1022789072, 1473955358]}),
        )
        for seed, schedule in expected:
            self.assertEqual(sampling.schedule(seed, (1, 2, 3)), schedule)
            full = sampling.schedule(seed, range(1, 721))
            self.assertEqual(len(set(full["seeds"])), 720 if schedule["mode"] == "per-frame" else 1)
            self.assertEqual(sampling.schedule(seed, range(301, 361))["seeds"], full["seeds"][300:360])
            self.assertTrue(all(0 <= value <= 2147483647 for value in full["seeds"]))
        modes = [sampling.schedule(seed, (1,))["mode"] for seed in range(4096)]
        self.assertEqual(set(modes), {"per-frame", "fixed"})
        self.assertGreater(modes.count("per-frame"), 1900)
        self.assertLess(modes.count("per-frame"), 2200)
        self.assertNotEqual(sampling.schedule(0, (1,)), sampling.schedule(1, (1,)))

    def test_composed_usd_and_manifest_carry_each_seed_without_raising_spp(self) -> None:
        default_spp = cli.parser().parse_args(["build", "--root", "."]).samples
        self.assertEqual(default_spp, 64)
        cases = ((0, False, False, 64), (2, True, False, 128), (0, False, True, 128), (2, True, True, 128))
        for seed, dof, blur, spp in cases:
            with self.subTest(seed=seed, dof=dof, blur=blur):
                root = Path(self.enterContext(TemporaryDirectory()))
                lib = library()
                hero_files(root, lib)
                spec = ShotSpec(
                    seed,
                    frame_count=3,
                    foliage_density=0,
                    thin=(),
                    camera=CameraChoices(move="locked", depth_of_field=dof, motion_blur=blur),
                )
                shot = compose.plan(spec, lib, root)
                render = compose.RenderParams(4, 4, default_spp)
                compose.write(shot, lib, root, render)
                export.write(shot, lib, render)
                manifest = cast(
                    "dict[str, object]", json.loads((shot.directory / "manifest.json").read_text())
                )
                camera = cast("dict[str, object]", json.loads((shot.directory / "camera.json").read_text()))
                metadata = cast("dict[str, object]", manifest["render"])
                expected = sampling.schedule(seed, shot.frames)
                self.assertEqual(records.errors(manifest, camera), [])
                self.assertEqual(metadata["sampling_seed"], expected)
                self.assertEqual(metadata["path_traced_samples"], spp)
                schema = {**validate.SCHEMA, "$ref": "#/$defs/manifest"}
                self.assertEqual(list(Draft202012Validator(schema).iter_errors(manifest)), [])
                stage = Usd.Stage.Open(str(shot.scene_dir / "shot.usda"))
                prim = stage.GetPrimAtPath(settings.SETTINGS_PATH)
                attr = prim.GetAttribute("karma:global:randomseed")
                self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Int)
                self.assertEqual([attr.Get(frame) for frame in shot.frames], expected["seeds"])
                self.assertEqual(attr.GetTimeSamples(), [1.0, 2.0, 3.0] if seed == 2 else [])
                self.assertEqual(prim.GetAttribute("karma:global:pathtracedsamples").Get(), spp)
                self.assertEqual(pack.ShotFrames.load(shot.directory).shot_id, manifest["shot_id"])
                corrupted = copy.deepcopy(expected)
                corrupted["seeds"][-1] ^= 1
                metadata["sampling_seed"] = corrupted
                (shot.directory / "manifest.json").write_text(json.dumps(manifest))
                errors = validate.validate(shot.directory)
                self.assertTrue(errors)
                self.assertTrue(all("sampling_seed" in error for error in errors))
                del metadata["sampling_seed"]
                self.assertTrue(list(Draft202012Validator(schema).iter_errors(manifest)))
                (shot.directory / "manifest.json").write_text(json.dumps(manifest))
                refused = records.errors(manifest, camera)
                self.assertTrue(refused)
                with self.assertRaisesRegex(ValueError, "sampling_seed"):
                    pack.pack(shot.directory, range(1, 4))
                self.assertEqual(checks.check_shot(shot.directory)["failures"], refused)
                errors = validate.validate(shot.directory)
                self.assertTrue(errors)
                self.assertTrue(all("sampling_seed" in error for error in errors))

    def test_karma_default_requires_unauthored_settings_evidence(self) -> None:
        root = Path(self.enterContext(TemporaryDirectory()))
        lib = library()
        hero_files(root, lib)
        shot = compose.plan(ShotSpec(0, frame_count=3, foliage_density=0, thin=()), lib, root)
        render = compose.RenderParams(4, 4, 64)
        compose.write(shot, lib, root, render)
        export.write(shot, lib, render)
        path = shot.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        camera = json.loads((shot.directory / "camera.json").read_text())
        manifest["render"]["sampling_seed"] = {"mode": "karma_default"}
        schema = {**validate.SCHEMA, "$ref": "#/$defs/manifest"}
        for version in (None, "render-settings-0.4.1", "render-settings-0.5.0"):
            manifest["generator_versions"]["settings"] = version
            path.write_text(json.dumps(manifest))
            self.assertTrue(records.errors(manifest, camera))
            self.assertTrue(list(Draft202012Validator(schema).iter_errors(manifest)))
            with self.assertRaisesRegex(ValueError, "sampling_seed"):
                pack.pack(shot.directory, range(1, 4))
            self.assertTrue(checks.check_shot(shot.directory)["failures"])
        manifest["generator_versions"]["settings"] = "render-settings-0.4.0"
        path.write_text(json.dumps(manifest))
        self.assertEqual(records.errors(manifest, camera), [])
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(manifest)), [])
        for value in (0, 17):
            stage = Usd.Stage.Open(str(shot.scene_dir / "shot.usda"))
            prim = stage.GetPrimAtPath(settings.SETTINGS_PATH)
            prim.GetAttribute("karma:global:randomseed").Set(value)
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, "authored USD randomseed"):
                pack.pack(shot.directory, range(1, 4))
            self.assertIn("authored USD randomseed", str(checks.check_shot(shot.directory)["failures"]))
        prim.GetAttribute("karma:global:randomseed").Clear()
        prim.GetAttribute("karma:global:randomseed").Set(0, Usd.TimeCode(2))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "authored USD randomseed"):
            pack.ShotFrames.load(shot.directory)
        prim.RemoveProperty("karma:global:randomseed")
        stage.GetRootLayer().Save()
        self.assertEqual(pack.ShotFrames.load(shot.directory).shot_id, manifest["shot_id"])
        self.assertEqual(settings.sampling_errors(shot.directory, manifest), [])
        invalid: tuple[dict[str, object], ...] = (
            {"mode": "karma_default", "seeds": []},
            {"mode": "karma_default", "seeds": [1]},
        )
        for value in invalid:
            manifest["render"]["sampling_seed"] = value
            self.assertTrue(records.errors(manifest, camera))
            self.assertTrue(list(Draft202012Validator(schema).iter_errors(manifest)))

    def test_karma_default_packs_and_passes_qa_without_optional_scene(self) -> None:
        root = Path(self.enterContext(TemporaryDirectory()))
        write_shot(root, [[[Sample(0.2, 0.5, 1), SKY], SKY]])
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["generator_versions"] = {"settings": "render-settings-0.4.0"}
        manifest["render"]["sampling_seed"] = {"mode": "karma_default"}
        path.write_text(json.dumps(manifest))
        pack.pack(root, range(1, 2))
        self.assertEqual(checks.check_shot(root)["failures"], [])
        self.assertEqual(checks.accept(root, range(1, 2))[0].failures, [])
        self.assertEqual(checks.check_shot(root)["failures"], [])

    def test_schema_rejects_malformed_sampling_fields(self) -> None:
        schema = {
            **validate.SCHEMA,
            "$ref": "#/$defs/manifest/properties/render/properties/sampling_seed",
        }
        validator = Draft202012Validator(schema)
        self.assertEqual(list(validator.iter_errors({"mode": "fixed", "seeds": [0, 2147483647]})), [])
        invalid: tuple[object, ...] = (
            {"mode": "random", "seeds": [1]},
            {"mode": "fixed", "seeds": []},
            {"mode": "fixed", "seeds": [-1]},
            {"mode": "fixed", "seeds": [2147483648]},
            {"mode": "fixed", "seeds": [True]},
            {"mode": "fixed"},
            {"seeds": [1]},
            {"mode": "fixed", "seeds": [1], "extra": 0},
            None,
        )
        for value in invalid:
            self.assertTrue(list(validator.iter_errors(value)), value)

    def test_record_validation_rejects_corrupted_sampling_metadata(self) -> None:
        camera: dict[str, object] = {
            "contract": records.VERSION,
            "frames": [{"frame": frame} for frame in (1, 2, 3)],
        }
        manifest: dict[str, object] = {
            "contract": records.VERSION,
            "seed": 2,
            "render": {"frames": [1, 3], "sampling_seed": sampling.schedule(2, (1, 2, 3))},
        }
        self.assertEqual(records.errors(manifest, camera), [])
        changes: tuple[dict[str, object], ...] = (
            {"mode": "fixed", "seeds": [727542660, 1022789072, 1473955358]},
            {"mode": "per-frame", "seeds": [727542660] * 3},
            {"mode": "per-frame", "seeds": [727542660, 1022789072]},
            {"mode": "per-frame", "seeds": [727542660, 1022789072, 1473955359]},
            {"mode": "per-frame", "seeds": [True, 1022789072, 1473955358]},
            {"mode": "per-frame", "seeds": [float(727542660), 1022789072, 1473955358]},
            {"mode": "per-frame", "seeds": [-1, 1022789072, 1473955358]},
            {"mode": "per-frame", "seeds": [2147483648, 1022789072, 1473955358]},
            {"mode": "random", "seeds": [727542660, 1022789072, 1473955358]},
            {},
        )
        for change in changes:
            damaged = copy.deepcopy(manifest)
            cast("dict[str, object]", damaged["render"])["sampling_seed"] = change
            self.assertTrue(records.errors(damaged, camera), change)
        for frames in ([1, 2], [3, 1], [1], [1, "3"], [True, 3]):
            damaged = copy.deepcopy(manifest)
            cast("dict[str, object]", damaged["render"])["frames"] = frames
            self.assertTrue(records.errors(damaged, camera), frames)
        empty_range = copy.deepcopy(manifest)
        empty_range["render"] = {"frames": [2, 1], "sampling_seed": sampling.schedule(2, ())}
        self.assertTrue(records.errors(empty_range, {"contract": records.VERSION, "frames": []}))
        for seed in (True, False, 2.0):
            damaged = copy.deepcopy(manifest)
            damaged["seed"] = seed
            cast("dict[str, object]", damaged["render"])["sampling_seed"] = sampling.schedule(
                cast("int", seed), (1, 2, 3)
            )
            self.assertTrue(records.errors(damaged, camera), seed)
        damaged = copy.deepcopy(manifest)
        damaged["seed"] = 0
        cast("dict[str, object]", damaged["render"])["sampling_seed"] = {
            "mode": "fixed",
            "seeds": [1427725052] * 3,
        }
        self.assertTrue(records.errors(damaged, camera))
        del cast("dict[str, object]", manifest["render"])["sampling_seed"]
        self.assertTrue(records.errors(manifest, camera))


if __name__ == "__main__":
    unittest.main()
