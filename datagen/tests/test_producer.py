import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import OpenEXR
from jsonschema import Draft202012Validator

import validate
from strata_datagen import cli
from strata_datagen.exr import Deep, read_deep, read_flat
from strata_datagen.generators import lighting
from strata_datagen.qa import checks
from strata_datagen.render import pack
from strata_dataset import identity, roles
from strata_dataset import samples as contract_samples
from tests.concurrency import free
from tests.fixtures import NAMES, SKY, Sample, frame_record, merged, write_beauty, write_deep, write_shot

CLEAN: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]
SKY_IN_FRONT: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 0), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]


def shot_failures(shot: Path) -> list[str]:
    return cast("list[str]", checks.check_shot(shot)["failures"])


class RawEvidenceTest(unittest.TestCase):
    def test_a_corrupt_second_frame_is_rejected_and_keeps_its_raw_deep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "shots" / "s"
            write_shot(shot, [CLEAN, SKY_IN_FRONT])
            self.assertFalse(cli.finish_task(shot, range(1, 3)))
            self.assertTrue((shot / "deep" / "0002.exr").exists())
            self.assertTrue(pack.ShotFrames.load(shot).sample_dir(shot, 2).is_dir())
            self.assertFalse(checks.check_shot(shot)["passed"])
            self.assertTrue(any("frame 2: sky_in_front" in f for f in shot_failures(shot)))

    def test_deep_roles_contradicting_the_beauty_role_aovs_are_rejected_and_keep_raw_deep(self) -> None:
        unhero = [[Sample(0.5, 1.0, 2, roles=(1.0, 0.0, 0.0)), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]
        furred = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1, roles=(1.0, 1.0, 1.0)), SKY], SKY]
        for corrupt in (unhero, furred):
            for frame in (1, 2):
                with self.subTest(corrupt=corrupt, frame=frame), tempfile.TemporaryDirectory() as tmp:
                    shot = Path(tmp) / "s"
                    write_shot(shot, [CLEAN, CLEAN])
                    write_deep(shot / "deep" / f"{frame:04d}.exr", corrupt)
                    raw = shot / "deep" / f"{frame:04d}.exr"
                    self.assertFalse(cli.finish_task(shot, range(1, 3)))
                    self.assertTrue(raw.exists())
                    self.assertFalse(json.loads((shot / "qa" / f"{frame:04d}.json").read_text())["passed"])
                    self.assertTrue(raw.exists())
                    failures = shot_failures(shot)
                    self.assertTrue(any(f.startswith(f"frame {frame}: role_beauty_error") for f in failures))
                    self.assertFalse(any(f.startswith(f"frame {3 - frame}:") for f in failures), failures)

    def test_beauty_role_aovs_must_be_present_and_finite_and_agree_within_roundoff(self) -> None:
        for name in roles.AOVS:
            for change in ("drop", "nan", "roundoff", "off"):
                with self.subTest(name=name, change=change), tempfile.TemporaryDirectory() as tmp:
                    shot = Path(tmp) / "s"
                    write_shot(shot, [CLEAN, CLEAN])
                    _change_beauty_role(shot / "beauty" / "0002.exr", name, change)
                    passed = cli.finish_task(shot, range(1, 3))
                    self.assertEqual(passed, change == "roundoff")
                    self.assertEqual((shot / "deep" / "0002.exr").exists(), not passed)
                    frames = cast("list[dict[str, float]]", checks.check_shot(shot)["frames"])
                    self.assertEqual(frames[1]["merged_beauty_roles"], float(change not in ("drop", "nan")))

    def test_targets_are_checked_against_the_raw_deep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN])
            pack.pack(shot, range(1, 3))
            self.assertEqual(shot_failures(shot), [])
            sky = pack.targets(pack.samples(_sky_everywhere(shot)))
            pack.write_layers(sky, shot / "layers" / "0002.exr")
            failures = shot_failures(shot)
            self.assertTrue(any(f.startswith("frame 2: target_mismatch") for f in failures), failures)

    def test_changed_id_ranges_fail_raw_qa_and_keep_the_raw_deep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN])
            pack.pack(shot, range(1, 3))
            self.assertEqual(shot_failures(shot), [])
            _drop_hero_ids(shot)
            self.assertTrue((shot / "deep" / "0002.exr").exists())
            failures = shot_failures(shot)
            self.assertTrue(any(f.startswith("frame 2: valid_id_fraction=0.8") for f in failures), failures)

    def test_records_of_another_contract_version_or_id_domain_are_refused(self) -> None:
        wide = [
            {"name": "sky", "first_id": 0, "count": 1},
            {"name": "grass", "first_id": 1000, "count": 40000},
        ]
        changes: tuple[tuple[str, dict[str, object]], ...] = (
            ("manifest.json", {"contract": "strata-dataset-0.3.0"}),
            ("camera.json", {"contract": "strata-dataset-0.3.0"}),
            ("manifest.json", {"instance_ids": wide}),
        )
        for name, change in changes:
            with self.subTest(name=name, change=change), tempfile.TemporaryDirectory() as tmp:
                shot = Path(tmp) / "s"
                write_shot(shot, [CLEAN, CLEAN])
                record = json.loads((shot / name).read_text())
                (shot / name).write_text(json.dumps(record | change))
                for step in (pack.pack, checks.accept):
                    with self.assertRaisesRegex(
                        ValueError, r"not 'strata-dataset-0.5.0'|leaves \[0, 32768\)"
                    ):
                        step(shot, range(1, 3))
                self.assertFalse((shot / "layers").exists())
                self.assertFalse((shot / "qa").exists())
                self.assertFalse(checks.check_shot(shot)["passed"])
                self.assertTrue((shot / "deep" / "0002.exr").exists())


MERGES: tuple[tuple[dict[int, float], dict[int, str]], ...] = (
    ({1: 107 / 256, 3: 113 / 256, 4: 36 / 256}, {}),
    ({1: 0.5, 3: 0.5}, {}),
    ({1: 0.5, 2001: 0.5}, {2001: "rock"}),
    ({30000: 0.6, 30001: 0.4}, {30000: "grass", 30001: "grass"}),
    ({13456: 0.51, 13457: 0.49}, {13456: "hero_skin", 13457: "hero_groom"}),
    ({99: 0.99, 100: 0.01}, {99: "hero_skin", 100: "hero_hair"}),
    ({2: 0.99, 3: 0.01}, {}),
    ({2: 1.0 - 2.0**-17, 3: 2.0**-17}, {}),
)


class MergedSampleTest(unittest.TestCase):
    def test_merged_samples_keep_exact_roles_and_only_a_dominant_id(self) -> None:
        """Reviewed merges: the 1/3/4 mix read as 2, 1+3 as 2, 30000.4, dominant 99/1 and 51/49 pairs."""
        for sources, names in MERGES:
            with self.subTest(sources=sources), tempfile.TemporaryDirectory() as tmp:
                shot = Path(tmp) / "s"
                sample = merged(0.5, 1.0, sources, names=names)
                pixels: list[Sample | list[Sample]] = [[sample, SKY], [Sample(0.2, 0.5, 1), SKY], SKY]
                write_shot(shot, [pixels, pixels])
                manifest = json.loads((shot / "manifest.json").read_text())
                declared = {r["first_id"] for r in manifest["instance_ids"]}
                manifest["instance_ids"] += [
                    {"name": {**NAMES, **names}.get(i, f"layer_{i}"), "first_id": i, "count": 1}
                    for i in sorted(set(sources) - declared)
                ]
                (shot / "manifest.json").write_text(json.dumps(manifest))
                pack.pack(shot, range(1, 3))
                arrays = contract_samples.load(pack.ShotFrames.load(shot).sample_dir(shot, 1))
                finished = cli.finish_task(shot, range(1, 3))
                passed = bool(checks.check_shot(shot)["passed"])
                self.assertEqual(finished, passed)
                self.assertEqual((shot / "deep" / "0002.exr").exists(), not passed)
                for frame in (1, 2):
                    layers = read_flat(shot / "layers" / f"{frame:04d}.exr")
                    for role, value in zip(roles.ROLES, sample.role_values(), strict=True):
                        self.assertEqual(float(layers[f"roles.{role}"][0, 0]), float(np.float32(value)))
                    self.assertEqual(float(layers["roles.foreground"][0, 1]), 0.5)
                found = int(arrays["ids"][0])
                np.testing.assert_array_equal(arrays["roles"][0], np.array(sample.role_values(), np.float32))
                if found == identity.UNLABELLED:
                    self.assertFalse(passed)
                    continue
                self.assertIn(found, sources)
                self.assertLessEqual(1.0 - sources[found], 2.0 * identity.TOLERANCE)
                self.assertTrue(passed)


def _change_beauty_role(path: Path, name: str, change: str) -> None:
    """Drop one beauty role AOV, make a pixel of it NaN, or move it by 2^-17 or 2^-13."""
    part = OpenEXR.File(str(path), separate_channels=True).parts[0]
    header = dict(part.header)
    channels = {key: np.array(channel.pixels, np.float32) for key, channel in part.channels.items()}
    if change == "drop":
        del channels[name]
    else:
        step = {"nan": np.nan, "roundoff": 2.0**-17, "off": 2.0**-13}[change]
        channels[name][0, 1] += np.float32(step)
    OpenEXR.File(header, {k: OpenEXR.Channel(v) for k, v in channels.items()}).write(str(path))


def _drop_hero_ids(shot: Path) -> None:
    manifest = json.loads((shot / "manifest.json").read_text())
    manifest["instance_ids"] = [r for r in manifest["instance_ids"] if r["name"] != "hero_skin"]
    (shot / "manifest.json").write_text(json.dumps(manifest))


def _sky_everywhere(shot: Path) -> Deep:
    path = shot / "sky.exr"
    write_deep(path, [SKY, SKY, SKY])
    return read_deep(path)


class SkyTargetTest(unittest.TestCase):
    def test_sky_only_pixel_is_one_opaque_layer_with_sky_colour_and_no_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN])
            pack.pack(shot, range(1, 2))
            layers = read_flat(shot / "layers" / "0001.exr")
            np.testing.assert_allclose(layers["L0.A"][0], [1.0, 0.5, 1.0])
            np.testing.assert_allclose(layers["L0.R"][0, 2], 0.3, rtol=1e-3)
            np.testing.assert_allclose(layers["L0.invdepth"][0], [0.5, 0.5, 0.0])
            np.testing.assert_allclose(layers["L1.A"][0], [0.0, 1.0, 0.0])
            np.testing.assert_allclose(layers["L1.invdepth"][0], [0.0, 0.0, 0.0])
            for k in (2, 3):
                self.assertEqual(float(layers[f"L{k}.A"].max()), 0.0)
            metrics = checks.target_metrics(
                shot / "layers" / "0001.exr", read_flat(shot / "beauty" / "0001.exr")
            )
            self.assertEqual(metrics["target_format"], 1.0)
            self.assertLess(metrics["target_error"], 1e-6)

    def test_near_opaque_front_sample_keeps_the_visible_layer_behind_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.exr"
            write_deep(path, [[Sample(0.9999, 0.9999, 2, 1.0), Sample(1.0, 1.0, 1, 10.0)]])
            deep = read_deep(path)
            self.assertEqual(float(deep.channels["A"][0]), 1.0)
            layers = pack.layers(pack.samples(deep))
            self.assertAlmostEqual(float(layers.alpha[0, 0, 0]), 0.9999, places=6)
            self.assertAlmostEqual(float(layers.alpha[1, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(layers.inverse_depth[1, 0, 0]), 0.1, places=6)


HAIRSTYLE: list[Sample | list[Sample]] = [
    [Sample(0.1, 0.5, 4, 1.9), Sample(0.5, 1.0, 2), SKY],
    [Sample(0.2, 0.5, 1), SKY],
    SKY,
]


class RoleTest(unittest.TestCase):
    def test_every_frame_keeps_role_coverage_from_its_own_deep_after_finishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, HAIRSTYLE, CLEAN])
            manifest = json.loads((shot / "manifest.json").read_text())
            manifest["instance_ids"].append({"name": "hero_hair", "first_id": 4, "count": 1})
            (shot / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(cli.finish_task(shot, range(1, 4)))
            self.assertFalse((shot / "deep" / "0002.exr").exists())
            expected = {
                1: {"foreground": [1.0, 0.5, 0.0], "hero": [1.0, 0.0, 0.0], "fur": [0.0, 0.0, 0.0]},
                2: {"foreground": [1.0, 0.5, 0.0], "hero": [1.0, 0.0, 0.0], "fur": [0.5, 0.0, 0.0]},
            }
            for frame, wanted in ((1, expected[1]), (2, expected[2]), (3, expected[1])):
                targets = read_flat(shot / "layers" / f"{frame:04d}.exr")
                for role, values in wanted.items():
                    np.testing.assert_allclose(targets[f"roles.{role}"][0], values, atol=1e-6)
            self.assertEqual(shot_failures(shot), [])

    def test_targets_without_role_coverage_fail_raw_qa_on_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN])
            self.assertTrue(cli.finish_task(shot, range(1, 3)))
            target = shot / "layers" / "0002.exr"
            frame = read_flat(target)
            pack.write_layers({k: v for k, v in frame.items() if not k.startswith("roles.")}, target)
            failures = shot_failures(shot)
            self.assertIn("frame 2: role_coverage=0 outside 1.0..1.0", failures)
            self.assertTrue(any(f.startswith("frame 2: rebuilt_mismatch") for f in failures))


class SidecarTest(unittest.TestCase):
    def test_every_frame_publishes_contract_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN, CLEAN])
            pack.pack(shot, range(1, 4))
            names = sorted(p.name for p in (shot / "samples").iterdir())
            self.assertEqual(names, ["s_0001", "s_0002", "s_0003"])
            directory = shot / "samples" / "s_0001"
            self.assertEqual(contract_samples.errors(directory, (1, 3)), [])
            arrays = contract_samples.load(directory)
            floats = {k: np.asarray(v, np.float64) for k, v in arrays.items()}
            self.assertEqual(arrays["offsets"].tolist(), [0, 2, 4, 5])
            self.assertEqual(arrays["ids"].tolist(), [2, 0, 1, 0, 0])
            np.testing.assert_allclose(floats["z"], [2.0, 3000.0, 2.0, 3000.0, 3000.0])
            np.testing.assert_allclose(floats["alpha"], [1.0, 1.0, 0.5, 1.0, 1.0])
            np.testing.assert_array_equal(
                floats["roles"], [[1, 1, 0], [0, 0, 0], [0.5, 0, 0], [0, 0, 0], [0, 0, 0]]
            )
            self.assertEqual(arrays["noisy"].shape, (3, 1, 3))
            np.testing.assert_allclose(floats["denoised"], floats["noisy"] * 0.9, rtol=1e-6)
            for name in ("noisy.npy", "denoised.npy", "deep/z.npy", "deep/rgb.npy", "deep/ids.npy"):
                self.assertTrue(
                    np.load(directory / name, mmap_mode="r", allow_pickle=False).flags.c_contiguous
                )

    def test_a_sidecar_that_differs_from_its_raw_deep_fails_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN])
            pack.pack(shot, range(1, 3))
            pack.remove_samples(shot, range(1, 2))
            frame = pack.samples(read_deep(shot / "deep" / "0001.exr"))
            wrong = dataclasses.replace(frame, ids=np.zeros_like(frame.ids))
            published = shot / "samples" / "s_0001"
            pack.write_samples(wrong, shot / "beauty" / "0001.exr", published)
            self.assertEqual(contract_samples.errors(published, (1, 3)), [])
            failures = shot_failures(shot)
            self.assertTrue(any(f.startswith("frame 1: samples_mismatch") for f in failures), failures)
            pack.remove_samples(shot, range(1, 2))
            unnested = frame.roles.copy()
            unnested[2, 2] = 0.25
            pack.write_samples(
                dataclasses.replace(frame, roles=unnested), shot / "beauty" / "0001.exr", published
            )
            self.assertEqual(
                contract_samples.errors(published, (1, 3)),
                ["s_0001: roles must satisfy 0 <= fur <= hero <= foreground <= alpha"],
            )
            shutil.rmtree(shot / "samples")
            self.assertTrue(any("frame 1: s_0001: missing" in f for f in shot_failures(shot)))

    def test_a_sidecar_file_that_differs_from_its_recorded_identity_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN])
            pack.pack(shot, range(1, 2))
            ids = shot / "samples" / "s_0001" / "deep" / "ids.npy"
            np.save(ids, np.zeros_like(np.load(ids)), allow_pickle=False)
            errors = contract_samples.errors(ids.parent.parent, (1, 3))
            self.assertEqual(errors, ["s_0001: unreadable: deep/ids.npy does not match identity.json"])
            with self.assertRaisesRegex(ValueError, "not a valid published frame"):
                pack.pack(shot, range(1, 2))

    def test_an_interrupted_publish_publishes_nothing_and_a_retry_publishes(self) -> None:
        for name in ("write_bytes", "rename"):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                shot = Path(tmp) / "s"
                write_shot(shot, [CLEAN])
                with (
                    mock.patch.object(Path, name, side_effect=OSError("injected")),
                    self.assertRaises(OSError),
                ):
                    pack.pack(shot, range(1, 2))
                self.assertFalse((shot / "samples" / "s_0001").exists())
                self.assertEqual(list((shot / pack.STAGING).iterdir()), [])
                self.assertTrue(free(shot))
                pack.pack(shot, range(1, 2))
                self.assertEqual(contract_samples.errors(shot / "samples" / "s_0001", (1, 3)), [])

    def test_anything_but_the_identical_frame_at_the_published_path_is_refused(self) -> None:
        for kind in ("empty", "symlink", "file"):
            with self.subTest(kind), tempfile.TemporaryDirectory() as tmp:
                shot = Path(tmp) / "s"
                write_shot(shot, [CLEAN])
                pack.pack(shot, range(1, 2))
                published = shot / "samples" / "s_0001"
                elsewhere = published.rename(shot / "elsewhere")
                if kind == "empty":
                    published.mkdir()
                elif kind == "symlink":
                    published.symlink_to(Path("..") / elsewhere.name, target_is_directory=True)
                else:
                    published.write_bytes(b"")
                self.assertTrue(contract_samples.errors(published, (1, 3)))
                with self.assertRaisesRegex(ValueError, "not a valid published frame"):
                    pack.pack(shot, range(1, 2))
                self.assertEqual(pack.remove_samples(shot, range(1, 2)), [1])
                pack.pack(shot, range(1, 2))
                self.assertEqual(contract_samples.errors(published, (1, 3)), [])


class SchemaTest(unittest.TestCase):
    def test_the_manifest_sky_id_range_is_valid(self) -> None:
        schema = {**validate.SCHEMA, "$ref": "#/$defs/id_range"}
        sky = {"name": "sky", "first_id": lighting.SKY_ID, "count": 1}
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(sky)), [])


class IntrinsicsTest(unittest.TestCase):
    def test_recorded_intrinsics_must_match_the_rendered_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN])
            pack.pack(shot, range(1, 2))
            beauty = shot / "beauty" / "0001.exr"
            write_beauty(beauty, read_deep(shot / "deep" / "0001.exr"), fy_scale=1.5)
            failures = shot_failures(shot)
            self.assertTrue(any(f.startswith("frame 1: intrinsics_error") for f in failures), failures)

    def test_fy_must_equal_fx_for_square_pixels(self) -> None:
        record = frame_record(1, 640, 640)
        record["fy"] = 50.0 / 20.25 * 640
        failures = checks.camera_consistency({"frames": [record]})
        self.assertIn("frame 1: fy differs from fx with square pixels", failures)


if __name__ == "__main__":
    unittest.main()
