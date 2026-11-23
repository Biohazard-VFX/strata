import dataclasses
import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jsonschema import Draft202012Validator

import validate
from strata_datagen import cli
from strata_datagen.qa import checks
from strata_datagen.render import pack
from strata_datagen.scene import compose
from strata_dataset import records
from tests.fixtures import SKY, Sample, write_shot

PIXELS: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]


class TemporalDeepTest(unittest.TestCase):
    def test_finishing_retains_consecutive_raw_deep_and_both_sidecars(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS] * 3)
            before = {path.name: path.read_bytes() for path in (shot / "deep").iterdir()}
            self.assertTrue(cli.finish_task(shot, range(1, 4)))
            self.assertEqual({path.name: path.read_bytes() for path in (shot / "deep").iterdir()}, before)
            plan = pack.ShotFrames.load(shot)
            for frame in range(1, 4):
                self.assertTrue((plan.merged_dir(shot, frame) / "deep.npz").is_file())
                self.assertTrue((plan.sample_dir(shot, frame) / "deep" / "z.npy").is_file())
            self.assertEqual([result.failures for result in checks.accept(shot, range(1, 4))], [[], [], []])

    def test_missing_middle_frame_deep_is_invalid_despite_a_passed_receipt(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(validate, "_schema_errors", return_value=[]),
            mock.patch.object(validate, "_beauty_errors", return_value=[]),
        ):
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS] * 3)
            self.assertTrue(cli.finish_task(shot, range(1, 4)))
            self.assertEqual(validate.validate(shot), [])
            plan = pack.ShotFrames.load(shot)
            for path, expected in (
                (shot / "deep" / "0002.exr", "missing raw deep for frame 2"),
                (plan.merged_dir(shot, 2) / "deep.npz", "merged"),
            ):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.unlink()
                    self.assertTrue(any(expected in error for error in validate.validate(shot)))
                    self.assertFalse(checks.check_shot(shot)["passed"])
                    path.write_bytes(original)
            self.assertEqual(validate.validate(shot), [])

    def test_frame_skipping_manifest_fields_and_public_controls_are_rejected(self) -> None:
        render = {
            "renderer": "fixture",
            "width": 3,
            "height": 1,
            "path_traced_samples": 1,
            "frames": [1, 3],
            "aovs": [],
            "deep_aovs": [],
        }
        schema = {**validate.SCHEMA, "$ref": "#/$defs/manifest/properties/render"}
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(render)), [])
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS] * 3)
            path = shot / "manifest.json"
            manifest = json.loads(path.read_text())
            camera = json.loads((shot / "camera.json").read_text())
            for value in (None, 1, 2, 24):
                with self.subTest(value=value):
                    manifest["render"]["deep_stride"] = value
                    path.write_text(json.dumps(manifest))
                    self.assertTrue(records.errors(manifest, camera))
                    with self.assertRaisesRegex(ValueError, "deep_stride"):
                        pack.ShotFrames.load(shot)
                    self.assertTrue(
                        list(Draft202012Validator(schema).iter_errors(render | {"deep_stride": value}))
                    )
        self.assertNotIn("deep_stride", {field.name for field in dataclasses.fields(compose.RenderParams)})
        self.assertNotIn("stride", {field.name for field in dataclasses.fields(pack.ShotFrames)})
        self.assertFalse(hasattr(compose, "DEEP_STRIDE"))
        self.assertFalse(hasattr(pack, "DEEP_STRIDE"))
        for arguments in (
            ["build", "--root", ".", "--deep-stride", "1"],
            ["prune", "--shot", ".", "--first", "1", "--count", "3"],
        ):
            with (
                self.subTest(arguments=arguments),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli.parser().parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
