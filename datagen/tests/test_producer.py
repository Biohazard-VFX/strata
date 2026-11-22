import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
from jsonschema import Draft202012Validator

import validate
from strata_contract import samples as contract_samples
from strata_datagen import cli
from strata_datagen.exr import Deep, read_deep, read_flat
from strata_datagen.generators import lighting
from strata_datagen.qa import checks
from strata_datagen.render import husk, pack
from tests.fixtures import SKY, Sample, frame_record, write_beauty, write_deep, write_shot

CLEAN: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]
SKY_IN_FRONT: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 0), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]


def run_post(root: Path, job: husk.Job) -> bool:
    """Run a farm task's post commands the way jobs.sh chains them; False when one fails."""
    for command in job.post:
        try:
            cli.main([str(root / job.workdir) if a == "." else a for a in command[1:]])
        except SystemExit:
            return False
    return True


def shot_failures(shot: Path) -> list[str]:
    return cast("list[str]", checks.check_shot(shot)["failures"])


def farm_job(root: Path, shot: Path, frames: int) -> husk.Job:
    (job,) = husk.jobs(shot, root, tuple(range(1, frames + 1)), husk.CHUNK_FRAMES)
    return job


class RawEvidenceTest(unittest.TestCase):
    def test_a_corrupt_frame_off_the_stride_is_rejected_and_keeps_its_raw_deep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "shots" / "s"
            write_shot(shot, [CLEAN, SKY_IN_FRONT], stride=24)
            self.assertFalse(run_post(root, farm_job(root, shot, 2)))
            self.assertTrue((shot / "deep" / "0002.exr").exists())
            self.assertFalse(checks.check_shot(shot)["passed"])
            self.assertTrue(any("frame 2: sky_id_in_front" in f for f in shot_failures(shot)))
            with self.assertRaises(ValueError):
                checks.prune(shot, range(1, 3))
            self.assertTrue((shot / "deep" / "0002.exr").exists())

    def test_pruned_frames_need_their_acceptance_and_unchanged_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "shots" / "s"
            write_shot(shot, [CLEAN, CLEAN, CLEAN], stride=2)
            self.assertTrue(run_post(root, farm_job(root, shot, 3)))
            self.assertEqual(sorted(p.name for p in (shot / "deep").iterdir()), ["0001.exr", "0003.exr"])
            self.assertEqual(shot_failures(shot), [])
            self.assertEqual([r.failures for r in checks.accept(shot, range(1, 4))], [[], [], []])
            self.assertEqual(shot_failures(shot), [])
            target = shot / "layers" / "0002.exr"
            kept = target.read_bytes()
            target.unlink()
            self.assertIn("frame 2: missing layers", shot_failures(shot))
            pack.write_layers(pack.layers(pack.samples(_sky_everywhere(shot))), target)
            self.assertIn("frame 2: targets changed since raw acceptance", shot_failures(shot))
            target.write_bytes(kept)
            (shot / "qa" / "0002.json").unlink()
            self.assertIn("frame 2: no raw deep and no raw acceptance", shot_failures(shot))

    def test_targets_are_checked_against_the_raw_deep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN], stride=24)
            pack.pack(shot, range(1, 3))
            self.assertEqual(shot_failures(shot), [])
            pack.write_layers(pack.layers(pack.samples(_sky_everywhere(shot))), shot / "layers" / "0002.exr")
            failures = shot_failures(shot)
            self.assertTrue(any(f.startswith("frame 2: target_mismatch") for f in failures), failures)


def _sky_everywhere(shot: Path) -> Deep:
    path = shot / "sky.exr"
    write_deep(path, [SKY, SKY, SKY])
    return read_deep(path)


class SkyTargetTest(unittest.TestCase):
    def test_sky_only_pixel_is_one_opaque_layer_with_sky_colour_and_no_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN], stride=1)
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


class SidecarTest(unittest.TestCase):
    def test_recomposition_frames_publish_contract_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN, CLEAN], stride=2)
            pack.pack(shot, range(1, 4))
            names = sorted(p.name for p in (shot / "samples").iterdir())
            self.assertEqual(names, ["s_0001", "s_0003"])
            directory = shot / "samples" / "s_0001"
            self.assertEqual(contract_samples.errors(directory, (1, 3)), [])
            arrays = contract_samples.load(directory)
            floats = {k: np.asarray(v, np.float64) for k, v in arrays.items()}
            self.assertEqual(arrays["offsets"].tolist(), [0, 2, 4, 5])
            self.assertEqual(arrays["ids"].tolist(), [2, 0, 1, 0, 0])
            np.testing.assert_allclose(floats["z"], [2.0, 3000.0, 2.0, 3000.0, 3000.0])
            np.testing.assert_allclose(floats["alpha"], [1.0, 1.0, 0.5, 1.0, 1.0])
            self.assertEqual(arrays["noisy"].shape, (3, 1, 3))
            np.testing.assert_allclose(floats["denoised"], floats["noisy"] * 0.9, rtol=1e-6)
            for name in ("noisy.npy", "denoised.npy", "deep/z.npy", "deep/rgb.npy", "deep/ids.npy"):
                self.assertTrue(
                    np.load(directory / name, mmap_mode="r", allow_pickle=False).flags.c_contiguous
                )

    def test_a_sidecar_that_differs_from_its_raw_deep_fails_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN, CLEAN], stride=24)
            pack.pack(shot, range(1, 3))
            ids = shot / "samples" / "s_0001" / "deep" / "ids.npy"
            np.save(ids, np.zeros_like(np.load(ids)), allow_pickle=False)
            failures = shot_failures(shot)
            self.assertTrue(any(f.startswith("frame 1: samples_mismatch") for f in failures), failures)
            shutil.rmtree(shot / "samples")
            self.assertTrue(any("frame 1: s_0001: missing" in f for f in shot_failures(shot)))


class SchemaTest(unittest.TestCase):
    def test_the_manifest_sky_id_range_is_valid(self) -> None:
        schema = {**validate.SCHEMA, "$ref": "#/$defs/id_range"}
        sky = {"name": "sky", "first_id": lighting.SKY_ID, "count": 1}
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(sky)), [])


class IntrinsicsTest(unittest.TestCase):
    def test_recorded_intrinsics_must_match_the_rendered_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s"
            write_shot(shot, [CLEAN], stride=1)
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
