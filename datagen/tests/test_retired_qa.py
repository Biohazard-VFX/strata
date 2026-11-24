import hashlib
import json
import unittest
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast
from unittest import mock

import numpy as np
import OpenEXR

from strata_datagen import cli
from strata_datagen.qa import checks
from strata_datagen.render import pack
from strata_dataset import merged
from tests.fixtures import SKY, Sample, write_shot
from tests.test_producer import CLEAN, SKY_IN_FRONT


class RendererCrop(TypedDict):
    frame: int
    counts: list[int]
    deep: dict[str, list[float]]
    beauty: dict[str, list[float]]


def write_renderer_crop(shot: Path, crop: RendererCrop) -> None:
    write_shot(shot, [[SKY] * len(crop["counts"])])
    path = shot / "deep" / "0001.exr"
    header = dict(OpenEXR.File(str(path)).parts[0].header)
    offsets = np.r_[0, np.cumsum(crop["counts"])]
    channels: dict[str, OpenEXR.Channel] = {}
    for name, samples in crop["deep"].items():
        cells = np.empty((1, len(crop["counts"])), object)
        values = np.asarray(samples, np.float32)
        for p, (first, last) in enumerate(pairwise(offsets)):
            cells[0, p] = values[first:last]
        channels[name] = OpenEXR.Channel(cells)
    OpenEXR.File(header, channels).write(str(path))
    path = shot / "beauty" / "0001.exr"
    part = OpenEXR.File(str(path), separate_channels=True).parts[0]
    for name in part.channels:
        part.channels[name].pixels = np.asarray(crop["beauty"][name], np.float32)[None]
    OpenEXR.File(dict(part.header), part.channels).write(str(path))


class RetiredQATest(unittest.TestCase):
    def test_wrong_merged_ids_keep_raw_even_with_identical_rebuilt_layers(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [CLEAN])
            pack.pack(shot, range(1, 2))
            plan = pack.ShotFrames.load(shot)
            path = plan.merged_dir(shot, 1) / "deep.npz"
            runs = merged.read(path)
            before = pack.layer_channels(runs.targets())
            runs.ids[0], runs.ids[2] = runs.ids[2], runs.ids[0]
            np.savez_compressed(path, allow_pickle=False, **runs.arrays())
            for name, value in before.items():
                np.testing.assert_array_equal(value, pack.layer_channels(merged.read(path).targets())[name])
            result = checks.accept(shot, range(1, 2))[0]
            self.assertEqual(result.metrics["rebuilt_mismatch"], 0.0)
            self.assertEqual(result.failures, ["merged_mismatch=1 outside 0.0..0.0"])
            self.assertTrue((shot / "deep" / "0001.exr").is_file())
            self.assertTrue(plan.sample_dir(shot, 1).is_dir())
            self.assertFalse(json.loads((shot / "qa" / "0001.json").read_text())["passed"])

    def test_retired_frame_runs_every_available_limit_fresh(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [CLEAN])
            self.assertTrue(cli.finish_task(shot, range(1, 2)))
            baseline = checks.check_shot(shot)
            self.assertTrue(baseline["passed"])
            metrics = cast("list[dict[str, float]]", baseline["frames"])[0]
            for metric in (
                "sky_in_front",
                "uncovered",
                "partial_share",
                "flatten_error",
                "flatten_excess",
                "valid_id_fraction",
                "unlabelled_coverage",
                "role_beauty_error",
                "beauty_roles",
            ):
                value = metrics[f"merged_{metric}"]
                with (
                    self.subTest(metric=metric),
                    mock.patch.dict(checks.LIMITS, {metric: (value + 1, value + 2)}),
                ):
                    result = checks.accept(shot, range(1, 2))[0]
                    self.assertTrue(
                        any(f"merged_{metric}=" in failure for failure in result.failures), result
                    )
            for metric in checks.RAW_ONLY_LIMITS:
                low, high = checks.LIMITS[metric]
                with self.subTest(metric=metric), mock.patch.dict(checks.LIMITS, {metric: (low, high + 1)}):
                    result = checks.accept(shot, range(1, 2))[0]
                    self.assertTrue(
                        any(
                            f"{metric}: raw QA policy changed; re-render required" in f
                            for f in result.failures
                        ),
                        result,
                    )
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_retired_compositing_metrics_measure_signed_bias_and_partial_uncovered_pixels(self) -> None:
        for bias in (-0.0625, 0.0625):
            with self.subTest(bias=bias), TemporaryDirectory() as temporary:
                shot = Path(temporary) / "shot"
                write_shot(shot, [[Sample(0.25, 0.5, 2), Sample(0.125, 0.25, 1), Sample(0.25, 0.5, 1)]])
                beauty = shot / "beauty" / "0001.exr"
                part = OpenEXR.File(str(beauty), separate_channels=True).parts[0]
                for name in "RGB":
                    part.channels[name].pixels = np.asarray(
                        part.channels[name].pixels, np.float32
                    ) + np.float32(bias)
                OpenEXR.File(dict(part.header), part.channels).write(str(beauty))
                pack.pack(shot, range(1, 2))
                relaxed = dict.fromkeys(
                    ("uncovered", "flatten_error", "flatten_excess", "target_error", "target_excess"),
                    (0.0, 1.0),
                )
                with mock.patch.dict(checks.LIMITS, relaxed):
                    self.assertEqual(checks.accept(shot, range(1, 2))[0].failures, [])
                result = checks.accept(shot, range(1, 2))[0]
                expected = {
                    "partial_share": 1.0,
                    "uncovered": 1.0,
                    "flatten_error": abs(bias),
                    "flatten_excess": max(-bias, 0.0),
                    "sky_in_front": 0.0,
                }
                for metric, value in expected.items():
                    self.assertAlmostEqual(result.metrics[f"merged_{metric}"], value, places=12)
                for metric in ("uncovered", "flatten_error"):
                    self.assertTrue(any(f"merged_{metric}=" in f for f in result.failures), result)
                self.assertEqual(any("merged_flatten_excess=" in f for f in result.failures), bias < 0)
                with mock.patch.dict(checks.LIMITS, {"partial_share": (0.0, 0.5)}):
                    self.assertTrue(
                        any(
                            "merged_partial_share=" in f for f in checks.accept(shot, range(1, 2))[0].failures
                        )
                    )

    def test_identity_and_sky_are_merge_invariant_including_threshold_crossing_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(
                shot,
                [
                    [
                        [Sample(0.1, 0.25, 0, 740), Sample(0.1, 0.5, 0, 760), SKY],
                        [
                            Sample(0.1, 0.5, 2, 1),
                            Sample(0.01, 0.01, 1, 2, checks=(0.0, 0.0)),
                            Sample(0.01, 0.01, 1, 2.1, checks=(0.0, 0.0)),
                            SKY,
                        ],
                        [Sample(0.2, 1.0, 2), Sample(0.1, 0.5, 0, 3), SKY],
                    ]
                ],
            )
            pack.pack(shot, range(1, 2))
            with mock.patch.dict(
                checks.LIMITS, {"sky_in_front": (0.0, 1.0), "unlabelled_coverage": (0.0, 1.0)}
            ):
                raw = checks.accept(shot, range(1, 2))[0]
                self.assertEqual(raw.failures, [])
                retired = checks.accept(shot, range(1, 2))[0]
                self.assertEqual(retired.failures, [])
            self.assertAlmostEqual(raw.metrics["sky_in_front"], 0.625 / (1.0 + 0.5 * 0.99**2), places=8)
            self.assertAlmostEqual(raw.metrics["unlabelled_coverage"], 0.5 * (1.0 - 0.99**2) / 3, places=8)
            for name in ("sky_in_front", "unlabelled_coverage"):
                self.assertAlmostEqual(raw.metrics[name], raw.metrics[f"merged_{name}"], places=14)
                self.assertAlmostEqual(raw.metrics[name], retired.metrics[f"merged_{name}"], places=14)

    def test_every_retained_limit_is_merge_invariant(self) -> None:
        crops = cast(
            "list[RendererCrop]",
            json.loads((Path(__file__).parent / "data" / "qa_renderer_crops.json").read_text()),
        )
        fixtures: list[RendererCrop | None] = [None, *crops]
        for crop in fixtures:
            with self.subTest(frame=crop["frame"] if crop else "adversarial"), TemporaryDirectory() as tmp:
                shot = Path(tmp) / "shot"
                if crop:
                    write_renderer_crop(shot, crop)
                else:
                    write_shot(
                        shot,
                        [
                            [
                                [Sample(0.1, 0.5, 2, 1), Sample(0.01, 0.01, 1, 2, checks=(0.0, 0.0)), SKY],
                                [Sample(0.01, 0.05, 7, 2 + i * 0.001) for i in range(10)] + [SKY],
                                [Sample(0.1, 0.3333, 0, 740), Sample(0.2, 0.5, 0, 760), SKY],
                                [Sample(0.1, 0.5, 1, 0.1), Sample(0.1, 1.0, 2, 0.14)],
                            ]
                        ],
                    )
                pack.pack(shot, range(1, 2))
                with mock.patch.dict(checks.LIMITS, dict.fromkeys(checks.LIMITS, (-1e10, 1e10))):
                    raw = checks.accept(shot, range(1, 2))[0]
                    self.assertEqual(raw.failures, [])
                    self.assertFalse((shot / "deep" / "0001.exr").exists())
                    retired = checks.accept(shot, range(1, 2))[0]
                    self.assertEqual(retired.failures, [])
                for name in checks.LIMITS.keys() - checks.RAW_ONLY_LIMITS.keys():
                    with self.subTest(metric=name):
                        key = f"merged_{name}" if f"merged_{name}" in retired.metrics else name
                        np.testing.assert_allclose(
                            raw.metrics[name], retired.metrics[key], rtol=1e-14, atol=1e-14
                        )
                for name in checks.RAW_ONLY_LIMITS:
                    self.assertNotIn(f"merged_{name}", retired.metrics)

    def test_undeclared_visible_coverage_limit_before_and_after_retirement(self) -> None:
        for alpha, hidden, passes in ((0.05, False, True), (0.1, False, False), (0.1, True, True)):
            with self.subTest(alpha=alpha, hidden=hidden), TemporaryDirectory() as tmp:
                shot = Path(tmp) / "shot"
                bad = [Sample(0.01, alpha, 7, 2 + i * 0.001) for i in range(10)] + [SKY]
                if hidden:
                    bad.insert(0, Sample(0.2, 1.0, 1, 1.0))
                good = [Sample(0.2, 0.5, 1), SKY]
                write_shot(shot, [[bad, *[good for _ in range(12000)]]])
                pack.pack(shot, range(1, 2))
                raw = checks.accept(shot, range(1, 2))[0]
                expected = 1.0 if hidden else 1.0 - (1.0 - (1.0 - float(np.float32(alpha))) ** 10) / 12001
                self.assertAlmostEqual(raw.metrics["valid_id_fraction"], expected, places=14)
                self.assertAlmostEqual(raw.metrics["merged_valid_id_fraction"], expected, places=14)
                if hidden:
                    self.assertEqual(raw.metrics["valid_id_fraction"], 1.0)
                    self.assertEqual(raw.metrics["merged_valid_id_fraction"], 1.0)
                self.assertEqual(not raw.failures, passes, raw)
                if not passes:
                    self.assertTrue(any(f.startswith("valid_id_fraction=") for f in raw.failures), raw)
                    with mock.patch.dict(checks.LIMITS, {"valid_id_fraction": (0.99, 1.0)}):
                        self.assertEqual(checks.accept(shot, range(1, 2))[0].failures, [])
                self.assertFalse((shot / "deep" / "0001.exr").exists())
                retired = checks.accept(shot, range(1, 2))[0]
                self.assertEqual(not retired.failures, passes, retired)
                if not passes:
                    self.assertTrue(
                        any(f.startswith("merged_valid_id_fraction=") for f in retired.failures), retired
                    )

    def test_bad_raw_back_depth_is_rejected_before_retirement(self) -> None:
        with TemporaryDirectory() as tmp:
            shot = Path(tmp) / "shot"
            write_shot(shot, [CLEAN])
            path = shot / "deep" / "0001.exr"
            part = OpenEXR.File(str(path), separate_channels=True).parts[0]
            part.channels["ZBack"].pixels[0, 0][0] = np.float32(1.9)
            OpenEXR.File(dict(part.header), part.channels).write(str(path))
            pack.pack(shot, range(1, 2))
            result = checks.accept(shot, range(1, 2))[0]
            self.assertEqual(result.metrics["bad_depth"], 1.0)
            self.assertIn("bad_depth=1 outside 0.0..0.0", result.failures)
            self.assertTrue(path.is_file())

    def test_raw_role_excess_cannot_be_hidden_by_merging_or_retirement(self) -> None:
        with TemporaryDirectory() as tmp:
            shot = Path(tmp) / "shot"
            pixel = [
                Sample(0.2, 0.5, 2, 2, roles=(1.01, 1, 1)),
                Sample(0.2, 0.5, 2, 2.001, roles=(0.98, 0.98, 0.98)),
                SKY,
            ]
            write_shot(shot, [[pixel, [Sample(0.2, 0.5, 1), SKY], SKY]])
            pack.pack(shot, range(1, 2))
            raw = checks.accept(shot, range(1, 2))[0]
            self.assertAlmostEqual(raw.metrics["role_excess"], 0.005, places=7)
            self.assertTrue(any(f.startswith("role_excess=") for f in raw.failures), raw)
            with mock.patch.dict(
                checks.LIMITS, {"role_excess": (0, 1), "role_beauty_error": (0, 1), "role_coverage": (0, 1)}
            ):
                self.assertEqual(checks.accept(shot, range(1, 2))[0].failures, [])
            retired = checks.accept(shot, range(1, 2))[0]
            self.assertIn("role_excess: raw QA policy changed; re-render required", retired.failures)
            self.assertNotIn("merged_role_excess", retired.metrics)
            self.assertNotIn("merged_bad_depth", retired.metrics)

    def test_proof_refuses_missing_retained_evidence_even_if_receipt_also_says_none(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [CLEAN])
            self.assertTrue(cli.finish_task(shot, range(1, 2)))
            path = shot / "qa" / "0001.json"
            receipt = json.loads(path.read_text())
            for key, folder in (("beauty", "beauty"), ("targets", "layers"), ("merged", "merged")):
                target = (
                    pack.ShotFrames.load(shot).merged_dir(shot, 1) / "deep.npz"
                    if key == "merged"
                    else shot / folder / "0001.exr"
                )
                data = target.read_bytes()
                target.unlink()
                broken = receipt | {"inputs": receipt["inputs"] | {key: None}}
                path.write_text(json.dumps(broken))
                self.assertTrue(checks.accept(shot, range(1, 2))[0].failures)
                target.write_bytes(data)
            path.write_text(json.dumps(receipt))
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_sky_accepted_under_old_policy_fails_fresh_retired_qa(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [SKY_IN_FRONT])
            pack.pack(shot, range(1, 2))
            with mock.patch.dict(checks.LIMITS, {"sky_in_front": (0.0, 1.0)}):
                self.assertEqual(checks.accept(shot, range(1, 2))[0].failures, [])
            self.assertFalse((shot / "deep" / "0001.exr").exists())
            self.assertTrue(
                any("merged_sky_in_front=" in f for f in checks.accept(shot, range(1, 2))[0].failures)
            )

    def test_malformed_receipts_cannot_authorize_retired_frames(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [CLEAN])
            self.assertTrue(cli.finish_task(shot, range(1, 2)))
            path = shot / "qa" / "0001.json"
            original = path.read_text()
            good = json.loads(original)
            receipts: tuple[object, ...] = (
                [],
                good | {"frame": 2},
                good | {"passed": False},
                good | {"failures": ["bad"]},
                good | {"inputs": []},
            )
            for receipt in receipts:
                with self.subTest(receipt=receipt):
                    path.write_text(json.dumps(receipt))
                    self.assertIn(
                        "missing raw deep without matching acceptance",
                        checks.accept(shot, range(1, 2))[0].failures,
                    )
            path.write_text("{")
            self.assertTrue(checks.accept(shot, range(1, 2))[0].failures)
            path.write_text(original)
            self.assertTrue(checks.check_shot(shot)["passed"])

    def test_inputs_changed_during_qa_never_authorize_raw_retirement(self) -> None:
        original = checks.check_frame
        for folder in ("deep", "beauty", "layers", "merged"):
            with self.subTest(folder=folder), TemporaryDirectory() as temporary:
                shot = Path(temporary) / "shot"
                write_shot(shot, [CLEAN])
                pack.pack(shot, range(1, 2))
                plan = pack.ShotFrames.load(shot)
                changed = (
                    plan.merged_dir(shot, 1) / "deep.npz"
                    if folder == "merged"
                    else shot / folder / "0001.exr"
                )

                def change_after_measurement(
                    root: Path,
                    frame: int,
                    record: dict[str, object],
                    ranges: list[tuple[int, int]],
                    target: Path = changed,
                ) -> checks.FrameResult:
                    result = original(root, frame, record, ranges)
                    self.assertEqual(result.failures, [])
                    with target.open("ab") as stream:
                        stream.write(b"changed")
                    return result

                with mock.patch.object(checks, "check_frame", change_after_measurement):
                    result = checks.accept(shot, range(1, 2))[0]
                self.assertEqual(result.failures, ["inputs changed during QA; retry required"])
                self.assertTrue((shot / "deep" / "0001.exr").is_file())
                self.assertTrue(plan.sample_dir(shot, 1).is_dir())
                self.assertFalse(json.loads((shot / "qa" / "0001.json").read_text())["passed"])

    def test_reappearing_raw_evidence_must_match_the_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [CLEAN])
            pack.pack(shot, range(1, 2))
            raw_path = shot / "deep" / "0001.exr"
            raw = raw_path.read_bytes()
            samples = pack.ShotFrames.load(shot).sample_dir(shot, 1)
            arrays = {p.relative_to(samples): p.read_bytes() for p in samples.rglob("*") if p.is_file()}
            self.assertEqual(checks.accept(shot, range(1, 2))[0].failures, [])
            for kind in ("deep", "samples"):
                with self.subTest(kind=kind):
                    for name, value in arrays.items():
                        target = samples / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(value)
                    if kind == "deep":
                        raw_path.write_bytes(raw + b"changed")
                    else:
                        identity = samples / "identity.json"
                        data = json.loads(identity.read_text())
                        data["extra"] = "changed"
                        identity.write_text(json.dumps(data))
                    result = checks.accept(shot, range(1, 2))[0]
                    if kind == "deep":
                        self.assertEqual(result.failures, [])
                        receipt = json.loads((shot / "qa" / "0001.json").read_text())
                        self.assertEqual(
                            receipt["inputs"]["deep"], hashlib.sha256(raw + b"changed").hexdigest()
                        )
                    else:
                        self.assertIn("missing raw deep without matching acceptance", result.failures)


if __name__ == "__main__":
    unittest.main()
