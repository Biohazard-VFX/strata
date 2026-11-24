import dataclasses
import io
import json
import os
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

import OpenEXR
from jsonschema import Draft202012Validator

import validate
from strata_datagen import cli
from strata_datagen.exr import read_deep
from strata_datagen.qa import checks
from strata_datagen.render import pack
from strata_datagen.scene import compose
from strata_dataset import merged, records
from tests.fixtures import SKY, Sample, write_shot

PIXELS: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]


class Interrupted(BaseException):
    pass


class TemporalDeepTest(unittest.TestCase):
    def test_raw_retirement_follows_durable_acceptance_and_reruns_are_identical(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS] * 3)
            pack.pack(shot, range(1, 4))
            plan = pack.ShotFrames.load(shot)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in shot.glob("merged/*/deep.npz")}
            synced: set[Path] = set()
            fsync, unlink = os.fsync, Path.unlink

            def sync(descriptor: int) -> None:
                synced.add(Path(f"/proc/self/fd/{descriptor}").readlink())
                fsync(descriptor)

            def remove(path: Path, *, missing_ok: bool = False) -> None:
                if path.parent == shot / "deep":
                    frame = int(path.stem)
                    receipt = shot / "qa" / f"{frame:04d}.json"
                    accepted = json.loads(receipt.read_text())
                    self.assertTrue(accepted["passed"])
                    self.assertEqual(accepted["inputs"], checks.inputs(shot, frame))
                    retained = [
                        receipt,
                        shot / "beauty" / path.name,
                        shot / "layers" / path.name,
                        plan.merged_dir(shot, frame) / "deep.npz",
                        shot / "manifest.json",
                        shot / "camera.json",
                    ]
                    required = {*retained, *(p.parent for p in retained), shot, shot / "merged"}
                    self.assertLessEqual(required, synced)
                unlink(path, missing_ok=missing_ok)

            with mock.patch.object(os, "fsync", sync), mock.patch.object(Path, "unlink", remove):
                self.assertEqual([r.failures for r in checks.accept(shot, range(1, 4))], [[], [], []])
            for frame in range(1, 4):
                self.assertFalse((shot / "deep" / f"{frame:04d}.exr").exists())
                self.assertFalse(plan.sample_dir(shot, frame).exists())
                self.assertTrue((plan.merged_dir(shot, frame) / "deep.npz").is_file())
            retained = {
                p: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in shot.rglob("*")
                if p.is_file() and not p.name.startswith(".")
            }
            self.assertTrue(cli.finish_task(shot, range(1, 4)))
            self.assertTrue(checks.check_shot(shot)["passed"])
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in retained}, retained)
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}, before)
            with self.assertRaisesRegex(ValueError, "cannot remove merged deep without raw"):
                pack.remove_samples(shot, range(1, 4))
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_interruptions_at_each_durability_and_deletion_boundary_recover(self) -> None:
        for operation, stops in (
            ("sync", range(1, 19)),
            ("receipt", range(1, 3)),
            ("raw", range(1, 3)),
            ("npy", range(1, 3)),
        ):
            for stop in stops:
                with self.subTest(operation=operation, stop=stop), TemporaryDirectory() as temporary:
                    self.interrupted_accept(Path(temporary) / "shot", operation, stop)

    def interrupted_accept(self, shot: Path, operation: str, stop: int) -> None:
        write_shot(shot, [PIXELS])
        pack.pack(shot, range(1, 2))
        plan = pack.ShotFrames.load(shot)
        raw = shot / "deep" / "0001.exr"
        count = 0
        fsync, replace, unlink, remove = os.fsync, Path.replace, Path.unlink, os.unlink

        def interrupt() -> None:
            nonlocal count
            count += 1
            if count == stop:
                raise Interrupted

        def sync(descriptor: int) -> None:
            interrupt()
            fsync(descriptor)

        def publish(path: Path, target: Path) -> Path:
            if operation == "receipt" and target.parent == shot / "qa":
                interrupt()
                result = replace(path, target)
                interrupt()
                return result
            return replace(path, target)

        def retire(path: Path, *, missing_ok: bool = False) -> None:
            if operation == "raw" and path == raw:
                interrupt()
                unlink(path, missing_ok=missing_ok)
                interrupt()
            else:
                unlink(path, missing_ok=missing_ok)

        def retire_array(path: str, *, dir_fd: int | None = None) -> None:
            if operation == "npy" and str(path).endswith(".npy"):
                interrupt()
                remove(path, dir_fd=dir_fd)
                interrupt()
            else:
                remove(path, dir_fd=dir_fd)

        with (
            mock.patch.object(os, "fsync", sync if operation == "sync" else fsync),
            mock.patch.object(Path, "replace", publish),
            mock.patch.object(Path, "unlink", retire),
            mock.patch.object(os, "unlink", retire_array),
            self.assertRaises(Interrupted),
        ):
            checks.accept(shot, range(1, 2))
        self.assertEqual(merged.errors(plan.merged_dir(shot, 1) / "deep.npz", (1, 3)), [])
        self.assertTrue(cli.finish_task(shot, range(1, 2)))
        self.assertFalse(raw.exists())
        self.assertFalse(plan.sample_dir(shot, 1).exists())
        self.assertTrue(checks.check_shot(shot)["passed"])

    def test_bookkeeping_replacement_is_durable_after_raw_retirement(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS])
            synced: set[Path] = set()
            fsync, replace = os.fsync, Path.replace

            def sync(descriptor: int) -> None:
                synced.add(Path(f"/proc/self/fd/{descriptor}").readlink())
                fsync(descriptor)

            def publish(path: Path, target: Path) -> Path:
                if target == shot / "manifest.json":
                    self.assertIn(path, synced)
                    self.assertFalse((shot / "deep" / "0001.exr").exists())
                return replace(path, target)

            with (
                mock.patch.object(os, "fsync", sync),
                mock.patch.object(Path, "replace", publish),
                mock.patch.object(cli.husk, "run", return_value={"exit_code": 0}),
            ):
                result = cli.render_task(shot, range(1, 2), "husk")
            self.assertTrue(result["post_passed"])
            self.assertEqual(json.loads((shot / "manifest.json").read_text())["timings"], result)
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_offline_removal_requires_durable_repackable_replacement_raw(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS])
            raw = shot / "deep" / "0001.exr"
            original = raw.read_bytes()
            self.assertTrue(cli.finish_task(shot, range(1, 2)))
            plan = pack.ShotFrames.load(shot)
            archive = plan.merged_dir(shot, 1) / "deep.npz"
            accepted = archive.read_bytes()
            for data in (b"", original[: len(original) // 2]):
                raw.write_bytes(data)
                with self.subTest(size=len(data)), self.assertRaises((ValueError, OSError, RuntimeError)):
                    pack.remove_samples(shot, range(1, 2))
                self.assertEqual(archive.read_bytes(), accepted)
            for channel in ("A", "ZBack"):
                raw.write_bytes(original)
                part = OpenEXR.File(str(raw), separate_channels=True).parts[0]
                del part.channels[channel]
                OpenEXR.File(dict(part.header), part.channels).write(str(raw))
                self.assertNotIn(channel, read_deep(raw).channels)
                with self.subTest(channel=channel), self.assertRaisesRegex(KeyError, channel):
                    pack.remove_samples(shot, range(1, 2))
                self.assertEqual(archive.read_bytes(), accepted)
            raw.write_bytes(original)
            with mock.patch.object(os, "fsync", side_effect=Interrupted), self.assertRaises(Interrupted):
                pack.remove_samples(shot, range(1, 2))
            self.assertEqual(archive.read_bytes(), accepted)
            synced: set[Path] = set()
            fsync, rename = os.fsync, Path.rename

            def sync(descriptor: int) -> None:
                synced.add(Path(f"/proc/self/fd/{descriptor}").readlink())
                fsync(descriptor)

            def retire(path: Path, target: Path) -> Path:
                self.assertLessEqual({raw, raw.parent}, synced)
                return rename(path, target)

            with mock.patch.object(os, "fsync", sync), mock.patch.object(Path, "rename", retire):
                self.assertEqual(pack.remove_samples(shot, range(1, 2)), [1])
            self.assertFalse(archive.exists())
            pack.pack(shot, range(1, 2))
            self.assertEqual([r.failures for r in checks.accept(shot, range(1, 2))], [[]])
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_rebuild_disagreement_keeps_raw_even_when_raw_qa_passes(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS])
            pack.pack(shot, range(1, 2))
            original = merged.Merged.targets

            def wrong(runs: merged.Merged, count: int = 4) -> object:
                result = original(runs, count)
                result.inverse_depth[0, 0, 0] += 0.1
                return result

            with mock.patch.object(merged.Merged, "targets", wrong):
                result = checks.accept(shot, range(1, 2))[0]
            self.assertIn("rebuilt_mismatch=1 outside 0.0..0.0", result.failures)
            self.assertTrue((shot / "deep" / "0001.exr").is_file())
            self.assertTrue(pack.ShotFrames.load(shot).sample_dir(shot, 1).is_dir())
            self.assertFalse(json.loads((shot / "qa" / "0001.json").read_text())["passed"])

    def test_merged_only_shot_requires_every_frame_and_matching_acceptance(self) -> None:
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
                (plan.merged_dir(shot, 2) / "deep.npz", "merged"),
                (shot / "qa" / "0002.json", "matching acceptance"),
            ):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.unlink()
                    if expected == "merged":
                        self.assertTrue(any(expected in error for error in validate.validate(shot)))
                    self.assertFalse(checks.check_shot(shot)["passed"])
                    path.write_bytes(original)
            self.assertEqual(validate.validate(shot), [])
            manifest_path = shot / "manifest.json"
            original = manifest_path.read_text()
            manifest = json.loads(original)
            manifest["instance_ids"] = [r for r in manifest["instance_ids"] if r["name"] != "hero_skin"]
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(checks.check_shot(shot)["passed"])
            manifest_path.write_text(original)
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_retired_proof_survives_qa_and_manifest_evolution(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS])
            self.assertTrue(cli.finish_task(shot, range(1, 2)))
            receipt = shot / "qa" / "0001.json"
            original = receipt.read_bytes()
            manifest_path = shot / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["new_metadata"] = {"value": 42}
            manifest_path.write_text(json.dumps(manifest))
            with (
                mock.patch.object(checks, "VALIDATION", "future-qa"),
                mock.patch.dict(checks.LIMITS, {"median_luma": (0.0, 2.0)}),
            ):
                report = checks.check_shot(shot)
                self.assertTrue(report["passed"])
                self.assertEqual(report["validation"], "future-qa")
                self.assertEqual(checks.accept(shot, range(1, 2))[0].failures, [])
            self.assertEqual(receipt.read_bytes(), original)
            with mock.patch.dict(checks.LIMITS, {"median_luma": (1.0, 2.0)}):
                failures = cast("list[str]", checks.check_shot(shot)["failures"])
                self.assertTrue(any("median_luma" in error for error in failures))
            with mock.patch.dict(checks.LIMITS, {"valid_id_fraction": (0.0, 0.5)}):
                failures = cast("list[str]", checks.check_shot(shot)["failures"])
                self.assertTrue(any("valid_id_fraction" in error for error in failures))

    def test_retired_proof_rejects_changed_retained_files_and_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS, [[Sample(0.5, 0.8, 2), SKY], [Sample(0.2, 0.4, 1), SKY], SKY]])
            self.assertTrue(cli.finish_task(shot, range(1, 3)))
            plan = pack.ShotFrames.load(shot)
            receipt = shot / "qa" / "0001.json"
            receipt_bytes = receipt.read_bytes()
            for folder in ("merged", "beauty", "layers"):
                path = (
                    plan.merged_dir(shot, 1) / "deep.npz"
                    if folder == "merged"
                    else shot / folder / "0001.exr"
                )
                other = (
                    plan.merged_dir(shot, 2) / "deep.npz"
                    if folder == "merged"
                    else shot / folder / "0002.exr"
                )
                original = path.read_bytes()
                for data in (original + b"changed", other.read_bytes()):
                    with self.subTest(folder=folder, swapped=data == other.read_bytes()):
                        path.write_bytes(data)
                        failures = cast("list[str]", checks.check_shot(shot)["failures"])
                        self.assertTrue(
                            any("missing raw deep without matching acceptance" in e for e in failures)
                        )
                        self.assertTrue(checks.accept(shot, range(1, 2))[0].failures)
                        self.assertEqual(receipt.read_bytes(), receipt_bytes)
                        path.write_bytes(original)
            data = json.loads(receipt_bytes)
            data["inputs"]["merged"] = "0" * 64
            receipt.write_text(json.dumps(data))
            self.assertFalse(checks.check_shot(shot)["passed"])
            receipt.unlink()
            self.assertFalse(checks.check_shot(shot)["passed"])
            receipt.write_bytes(receipt_bytes)
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_retired_algorithm_changes_require_rerender(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS])
            self.assertTrue(cli.finish_task(shot, range(1, 2)))
            for version in ("MERGE_VERSION", "REBUILD_VERSION"):
                with self.subTest(version=version), mock.patch.object(merged, version, "future"):
                    result = checks.accept(shot, range(1, 2))[0]
                    self.assertIn("merge/rebuild version changed; re-render required", result.failures)
                    self.assertFalse(checks.check_shot(shot)["passed"])
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_unlabelled_visible_coverage_gates_but_raw_count_is_diagnostic(self) -> None:
        for opacity in (1e-6, 0.016, 0.5):
            with self.subTest(opacity=opacity), TemporaryDirectory() as temporary:
                shot = Path(temporary) / "shot"
                pixel = [Sample(0.003, 0.01, 1 + i % 2) for i in range(10)] + [SKY]
                unlabelled = [Sample(0.0, opacity, 2, checks=(0.0, 0.0)) for _ in range(10)]
                write_shot(shot, [[unlabelled + pixel, *[pixel for _ in range(999)]]])
                pack.pack(shot, range(1, 2))
                result = checks.accept(shot, range(1, 2))[0]
                self.assertGreater(result.metrics["id_check_failure_fraction"], 2e-4)
                self.assertNotIn("merged_id_check_failure_fraction", result.metrics)
                self.assertAlmostEqual(
                    result.metrics["unlabelled_coverage"],
                    result.metrics["merged_unlabelled_coverage"],
                    places=14,
                )
                self.assertEqual(bool(result.failures), opacity == 0.5)
                self.assertEqual((shot / "deep" / "0001.exr").is_file(), opacity == 0.5)
                if opacity == 0.5:
                    self.assertTrue(all("unlabelled_coverage=" in f for f in result.failures), result)
                else:
                    self.assertTrue(checks.check_shot(shot)["passed"])
                    with mock.patch.dict(checks.LIMITS, {"unlabelled_coverage": (0.0, 1e-10)}):
                        self.assertTrue(
                            any(
                                "merged_unlabelled_coverage=" in f
                                for f in checks.accept(shot, range(1, 2))[0].failures
                            )
                        )

    def test_contract_validator_rejects_strided_camera_frames(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(validate, "_schema_errors", return_value=[]),
            mock.patch.object(validate, "_beauty_errors", return_value=[]),
        ):
            shot = Path(temporary) / "shot"
            write_shot(shot, [PIXELS] * 3)
            pack.pack(shot, range(1, 4))
            self.assertEqual(validate.validate(shot), [])
            path = shot / "camera.json"
            camera = json.loads(path.read_text())
            camera["frames"] = camera["frames"][::2]
            path.write_text(json.dumps(camera))
            self.assertIn("camera.json frames do not cover the rendered range 1..3", validate.validate(shot))

    def test_replaced_raw_invalidates_written_acceptance(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            replacement = Path(temporary) / "replacement"
            write_shot(shot, [PIXELS])
            write_shot(replacement, [[[Sample(0.8, 1.0, 2), SKY], SKY, SKY]])
            pack.pack(shot, range(1, 2))
            with (
                mock.patch.object(checks, "_retire_raw", side_effect=Interrupted),
                self.assertRaises(Interrupted),
            ):
                checks.accept(shot, range(1, 2))
            receipt = shot / "qa" / "0001.json"
            self.assertTrue(json.loads(receipt.read_text())["passed"])
            self.assertTrue(checks.check_shot(shot)["passed"])
            (shot / "deep" / "0001.exr").write_bytes((replacement / "deep" / "0001.exr").read_bytes())
            self.assertFalse(checks.check_shot(shot)["passed"])
            self.assertTrue(checks.accept(shot, range(1, 2))[0].failures)
            self.assertTrue((shot / "deep" / "0001.exr").is_file())
            self.assertTrue(pack.ShotFrames.load(shot).sample_dir(shot, 1).is_dir())

    def test_frame_skipping_manifest_fields_and_public_controls_are_rejected(self) -> None:
        render = {
            "renderer": "fixture",
            "width": 3,
            "height": 1,
            "path_traced_samples": 1,
            "sampling_seed": {"mode": "fixed", "seeds": [0, 0, 0]},
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
