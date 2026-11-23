import unittest
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from numpy.typing import NDArray

import validate
from strata_datagen.qa import checks
from strata_datagen.render import pack
from strata_dataset import merged, roles, segmented
from strata_dataset.layers import Layers, reduce_samples
from tests.fixtures import SKY, Sample, write_shot


def raw_targets(frame: pack.Samples, count: int) -> Layers:
    return reduce_samples(
        frame.z,
        frame.alpha,
        frame.rgb,
        np.diff(frame.offsets),
        frame.shape,
        layers=count,
        sky=roles.sky(frame.alpha, frame.roles[:, 0]),
    )


def fixture(seed: int, pixels: int = 1200) -> pack.Samples:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 65, pixels, dtype=np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(counts)]
    n = int(offsets[-1])
    steps = rng.choice([1.0, 1.03, 1.1, 1.15 - 2e-7, 1.15, 1.15 + 2e-7, 1.4, 2.0], n)
    z = np.empty(n, np.float32)
    for start, end in pairwise(offsets):
        z[start:end] = rng.uniform(0.5, 4) * np.cumprod(steps[start:end])
    alpha = rng.uniform(0.001, 0.8, n).astype(np.float32)
    alpha[rng.random(n) < 0.03] = 0
    alpha[rng.random(n) < 0.01] = 1
    alpha[rng.random(n) < 0.02] = np.float32(1e-12)
    rgb = rng.uniform(0, 8, (n, 3)).astype(np.float32) * alpha[:, None]
    semantic = np.sort(rng.uniform(0, 1, (n, 3)), axis=1)[:, ::-1].astype(np.float32) * alpha[:, None]
    semantic[rng.random(n) < 0.05] = 0
    ids = rng.integers(-1, 4, n, dtype=np.int64)
    ids[rng.random(n) < 0.7] = 1
    return pack.Samples((1, pixels), offsets, z, alpha, rgb, semantic, ids)


def flattened(
    alpha: NDArray[np.floating], values: NDArray[np.floating], counts: NDArray[np.int64]
) -> NDArray[np.float64]:
    return segmented.weighted_sum(segmented.transmittance(alpha, counts), values, counts)


@segmented.jit(cache=True)
def boundaries(
    z: NDArray[np.float32],
    back: NDArray[np.float32],
    visible: NDArray[np.float64],
    offsets: NDArray[np.int64],
    count: int,
) -> NDArray[np.float32]:
    output = np.zeros((offsets.size - 1, count - 1), np.float32)
    for pixel in range(offsets.size - 1):
        start, end = offsets[pixel], offsets[pixel + 1]
        kept = np.flatnonzero(visible[start:end] > 1e-12) + start
        gaps = np.log(z[kept[1:]].astype(np.float64)) - np.log(back[kept[:-1]].astype(np.float64))
        ranked = np.argsort(-gaps, kind="mergesort")
        chosen = ranked[gaps[ranked] > np.log(1.15)][: count - 1]
        chosen.sort()
        for slot in range(chosen.size):
            output[pixel, slot] = z[kept[chosen[slot] + 1]]
    return output


class MergedDeepTest(unittest.TestCase):
    def assert_targets(self, frame: pack.Samples, runs: merged.Merged) -> None:
        for count in range(1, 9):
            with self.subTest(layers=count):
                raw = raw_targets(frame, count)
                rebuilt = runs.targets(count)
                visible = segmented.transmittance(frame.alpha, np.diff(frame.offsets)) * frame.alpha
                np.testing.assert_array_equal(
                    boundaries(frame.z, frame.z, visible, frame.offsets, count),
                    boundaries(runs.z, runs.zback, runs.statistics[:, 1], runs.offsets, count),
                )
                np.testing.assert_array_equal(rebuilt.alpha > 0, raw.alpha > 0)
                np.testing.assert_array_equal(rebuilt.inverse_depth == 0, raw.inverse_depth == 0)
                np.testing.assert_allclose(rebuilt.alpha, raw.alpha, rtol=0, atol=1e-5)
                np.testing.assert_allclose(rebuilt.inverse_depth, raw.inverse_depth, rtol=0, atol=1e-5)
                np.testing.assert_allclose(rebuilt.rgb, raw.rgb, rtol=1e-5, atol=1e-12)

    def test_adjacent_chain_uses_back_depth_and_inverse_moment(self) -> None:
        frame = pack.Samples(
            (1, 1),
            np.array([0, 3], np.int64),
            np.array([1, 1.1, 1.2], np.float32),
            np.full(3, 0.4, np.float32),
            np.full((3, 3), 0.2, np.float32),
            np.full((3, 3), 0.4, np.float32),
            np.array([1, 1, 2], np.int64),
        )
        runs = merged.merge(
            frame.shape, frame.offsets, frame.z, frame.alpha, frame.rgb, semantic=frame.roles, ids=frame.ids
        )
        np.testing.assert_array_equal(runs.z, frame.z[[0, 2]])
        np.testing.assert_array_equal(runs.zback, frame.z[[1, 2]])
        np.testing.assert_array_equal(runs.ids, [1, 2])
        self.assert_targets(frame, runs)
        np.testing.assert_allclose(runs.alpha, [0.64, 0.4], atol=1e-7)
        np.testing.assert_allclose(runs.rgb[:, 0], [0.32, 0.2], atol=1e-7)

    def test_every_frame_publishes_and_qa_binds_merged_deep_before_pruning(self) -> None:
        with TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            pixels: list[Sample | list[Sample]] = [
                [Sample(0.5, 1.0, 2), SKY],
                [Sample(0.2, 0.5, 1), SKY],
                SKY,
            ]
            write_shot(shot, [pixels, pixels], stride=24)
            pack.pack(shot, range(1, 3))
            plan = pack.ShotFrames.load(shot)
            for frame in (1, 2):
                runs = merged.read(plan.merged_dir(shot, frame) / "deep.npz")
                self.assertEqual(runs.shape, (1, 3))
            self.assertFalse(plan.sample_dir(shot, 2).exists())
            self.assertEqual([result.failures for result in checks.accept(shot, range(1, 3))], [[], []])
            path = plan.merged_dir(shot, 2) / "deep.npz"
            good = path.read_bytes()
            path.unlink()
            self.assertTrue(checks.accept(shot, range(2, 3))[0].failures)
            path.write_bytes(good)
            checks.accept(shot, range(2, 3))
            corrupt = merged.read(path)
            corrupt.statistics[0, 2] += 1
            np.savez_compressed(path, allow_pickle=False, **corrupt.arrays())
            with self.assertRaisesRegex(ValueError, "merged"):
                checks.prune(shot, range(2, 3))
            self.assertTrue(checks.accept(shot, range(2, 3))[0].failures)
            path.write_bytes(good)
            self.assertEqual(checks.accept(shot, range(2, 3))[0].failures, [])
            self.assertEqual(checks.prune(shot, range(2, 3)), [2])
            path.unlink()
            self.assertFalse(checks.check_shot(shot)["passed"])
            self.assertEqual(pack.remove_samples(shot, range(1, 3)), [1, 2])
            self.assertFalse(plan.merged_dir(shot, 1).exists())

    def test_contract_validator_rejects_malformed_merged_sidecars(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            mock.patch.object(validate, "_schema_errors", return_value=[]),
            mock.patch.object(validate, "_beauty_errors", return_value=[]),
        ):
            shot = Path(temporary) / "shot"
            pixels: list[Sample | list[Sample]] = [
                [Sample(0.5, 1.0, 2), SKY],
                [Sample(0.2, 0.5, 1), SKY],
                SKY,
            ]
            write_shot(shot, [pixels, pixels], stride=24)
            pack.pack(shot, range(1, 3))
            self.assertEqual(validate.validate(shot), [])
            path = pack.ShotFrames.load(shot).merged_dir(shot, 2) / "deep.npz"
            arrays = merged.read(path).arrays()
            count = arrays["z"].size
            corruptions: list[tuple[str, NDArray[np.generic]]] = [
                ("shape", np.array([2, 3], np.int64)),
                ("shape", np.array([1, 3], np.float64)),
                ("offsets", np.array([0, -1, 1, count], np.int64)),
                ("z", arrays["z"].astype(np.float64)),
                ("alpha", np.full(count, 1.001, np.float64)),
                ("rgb", np.full((count, 3), np.nan, np.float64)),
                ("roles", np.full((count, 3), 2, np.float64)),
                ("ids", np.full(count, -2, np.int64)),
                ("zback", np.zeros(count, np.float32)),
                ("statistics", np.zeros((count, 3), np.float64)),
                ("statistics", np.full((count, 4), np.inf, np.float64)),
            ]
            for name, value in corruptions:
                with self.subTest(field=name, shape=value.shape, dtype=value.dtype):
                    np.savez_compressed(path, allow_pickle=False, **(arrays | {name: value}))
                    self.assertTrue(any("merged" in error for error in validate.validate(shot)))
            np.savez_compressed(
                path, allow_pickle=False, **{n: a for n, a in arrays.items() if n != "statistics"}
            )
            self.assertTrue(any("merged" in error for error in validate.validate(shot)))
            path.write_bytes(b"")
            self.assertTrue(any("merged" in error for error in validate.validate(shot)))
            with path.open("wb") as stream:
                np.save(stream, arrays["alpha"], allow_pickle=False)
            self.assertTrue(any("merged" in error for error in validate.validate(shot)))
            np.savez_compressed(path, allow_pickle=False, **arrays)
            self.assertEqual(validate.validate(shot), [])

    def test_seeded_multi_id_boundary_fuzz_roundtrip_and_flatten(self) -> None:
        for seed in (7, 71, 715):
            frame = fixture(seed)
            with self.subTest(seed=seed), TemporaryDirectory() as temporary:
                directory = Path(temporary) / "frame"
                pack.write_merged(frame, directory)
                runs = merged.read(directory / "deep.npz")
                self.assertLess(runs.z.size, frame.z.size)
                self.assert_targets(frame, runs)
                for original, packed in (
                    (frame.rgb, runs.rgb),
                    (frame.roles, runs.roles),
                    (frame.alpha[:, None], runs.alpha[:, None]),
                ):
                    np.testing.assert_allclose(
                        flattened(runs.alpha, packed, np.diff(runs.offsets)),
                        flattened(frame.alpha, original, np.diff(frame.offsets)),
                        rtol=1e-7,
                        atol=1e-10,
                    )
                before = (directory / "deep.npz").stat().st_mtime_ns
                pack.write_merged(frame, directory)
                self.assertEqual((directory / "deep.npz").stat().st_mtime_ns, before)
                frame.rgb[0, 0] += 1
                with self.assertRaisesRegex(ValueError, "different merged deep"):
                    pack.write_merged(frame, directory)


if __name__ == "__main__":
    unittest.main()
