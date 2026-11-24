import json
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import OpenEXR
from pxr import Usd, UsdGeom

from strata_datagen.exr import Deep, read_deep
from strata_datagen.generators import groom
from strata_datagen.library import hdri
from strata_datagen.qa import checks
from strata_datagen.qa.deep import instance_ids, partial_coverage
from strata_datagen.render import pack, settings
from strata_datagen.rng import stream
from strata_datagen.usdutil import relative_asset, tag_instances
from strata_dataset import identity, roles
from tests.fixtures import Sample, membership, write_shot
from tests.flatten import flatten

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _labels(
    sources: list[list[tuple[float, int]]], alpha: "NDArray[np.float32]"
) -> "dict[str, NDArray[np.float32]]":
    """Premultiplied check and role channels of samples merged from (share, id) sources."""
    out: dict[str, NDArray[np.float32]] = {}
    for name, value in (("instance_check", identity.check), ("instance_check2", identity.check2)):
        out[name] = (
            np.array([sum(w * float(value(np.array([i]))[0]) for w, i in s) for s in sources], np.float32)
            * alpha
        )
    for index, role in enumerate(roles.ROLES):
        out[role] = (
            np.array([sum(w * membership(i)[index] for w, i in s) for s in sources], np.float32) * alpha
        )
    return out


def _deep(samples: list[list[tuple[float, float, float, float, float, float]]]) -> Deep:
    """Pixels of (r, g, b, a, z, id) samples, unpremultiplied; stored the way Karma writes them."""
    counts = np.array([len(p) for p in samples])
    flat = [s for p in samples for s in p]
    a = np.array([s[3] for s in flat], np.float32)
    channels = {
        name: np.array([s[i] for s in flat], np.float32) * (a if i < 3 else 1.0)
        for i, name in enumerate("RGBA")
    }
    channels["Z"] = np.array([s[4] for s in flat], np.float32)
    channels["ZBack"] = channels["Z"].copy()
    ids = np.array([s[5] for s in flat], np.float64)
    channels["instance_id"] = ids.astype(np.float32) * a
    channels |= _labels([[(1.0, int(i))] for i in np.rint(ids)], a)
    channels["coverage"] = a.copy()
    channels["A"] = a.astype(np.float16).astype(np.float32)
    return Deep(len(samples), 1, counts, channels)


class DeepTest(unittest.TestCase):
    def test_partial_front_slab_includes_absolute_gap_and_opaque_alpha(self) -> None:
        depth = np.array([0.1, 0.14, 0.1, 0.16, 0.1, 0.1], np.float32)
        alpha = np.array([0.5, 1.0, 0.5, 1.0, 1.0, np.nextafter(1.0, 2.0)], np.float64)
        with np.errstate(all="raise"):
            found = partial_coverage(depth, alpha, np.array([2, 2, 1, 1]), np.ones(4))
        self.assertEqual(found.tolist(), [False, True, False, False])

    def test_flatten_composites_front_to_back_and_flags_see_through_pixels(self) -> None:
        deep = _deep(
            [
                [(0, 0, 1, 1.0, 5.0, 1000), (1, 0, 0, 0.5, 1.0, 7)],
                [(0, 1, 0, 1.0, 2.0, 2)],
                [(0, 1, 0, 0.3, 2.0, 25882)],
            ]
        )
        flat = flatten(deep)
        np.testing.assert_allclose(flat.rgba[0, 0], [0.5, 0.0, 0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(flat.rgba[0, 2], [0.0, 0.3, 0.0, 0.3], atol=2.0**-11)
        self.assertEqual(flat.partial[0].tolist(), [True, False, True])
        self.assertEqual(instance_ids(deep).tolist(), [1000, 7, 2, 25882])
        half_alpha = np.rint(deep.channels["instance_id"] / deep.channels["A"]).astype(int)
        self.assertNotEqual(half_alpha[-1], 25882)


def write_deep(path: Path, samples: list[list[tuple[float, ...]]]) -> None:
    """Pixels of (premultiplied grey, alpha, id[, depth, default 2 m]) samples in a 1-row deep image."""
    chans: dict[str, NDArray[np.object_]] = {}
    for name, index in (("R", 0), ("G", 0), ("B", 0), ("A", 1), ("coverage", 1), ("instance_id", 2)):
        cells = np.empty((1, len(samples)), dtype=object)
        for i, pixel in enumerate(samples):
            value = np.array([s[index] * (s[1] if name == "instance_id" else 1.0) for s in pixel], np.float32)
            cells[0, i] = value
        chans[name] = cells
    labels = [
        _labels([[(1.0, int(s[2]))] for s in p], np.array([s[1] for s in p], np.float32)) for p in samples
    ]
    for name in labels[0]:
        cells = np.empty((1, len(samples)), dtype=object)
        for i, found in enumerate(labels):
            cells[0, i] = found[name]
        chans[name] = cells
    for name in ("Z", "ZBack"):
        cells = np.empty((1, len(samples)), dtype=object)
        for i, pixel in enumerate(samples):
            cells[0, i] = np.array([s[3] if len(s) > 3 else 2.0 for s in pixel], np.float32)
        chans[name] = cells
    header: dict[str, object] = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.deepscanline}
    OpenEXR.File(header, {k: OpenEXR.Channel(v) for k, v in chans.items()}).write(str(path))


class QaTest(unittest.TestCase):
    def test_no_labelled_coverage_reports_fully_declared_and_fully_unlabelled(self) -> None:
        deep = _deep([[(0.2, 0.2, 0.2, 1.0, 2.0, 1)]])
        deep.channels["instance_check"][:] = 0.0
        beauty = {c: np.array([[0.2]], np.float32) for c in "RGB"}
        measured = checks.deep_metrics(deep, beauty, [(0, 4)])
        self.assertEqual(measured["valid_id_fraction"], 1.0)
        self.assertEqual(measured["unlabelled_coverage"], 1.0)

    def test_deep_with_a_sky_sample_flattens_to_the_beauty_everywhere(self) -> None:
        ranges = [(0, 4), (1000, 1001)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.exr"
            write_deep(path, [[(0.5, 1.0, 1000)], [(0.2, 0.5, 2), (0.6, 1.0, 0, 3000.0)]])
            beauty = {c: np.array([[0.5, 0.2 + 0.5 * 0.6]], np.float32) for c in "RGB"}
            metrics = checks.deep_metrics(read_deep(path), beauty, ranges)
            self.assertLess(metrics["flatten_error"], 1e-6)
            self.assertEqual(metrics["uncovered"], 0.0)
            self.assertEqual(metrics["valid_id_fraction"], 1.0)
            dark = {c: np.array([[0.5, 0.1]], np.float32) for c in "RGB"}
            self.assertGreater(checks.deep_metrics(read_deep(path), dark, ranges)["flatten_excess"], 0.04)
            write_deep(path, [[(0.5, 1.0, 1000)], [(0.2, 0.5, 2)]])
            self.assertEqual(checks.deep_metrics(read_deep(path), beauty, ranges)["uncovered"], 0.5)
            write_deep(path, [[(0.5, 1.0, 0)], [(0.2, 0.5, 2), (0.6, 1.0, 0, 3000.0)]])
            self.assertGreater(checks.deep_metrics(read_deep(path), beauty, ranges)["sky_in_front"], 0.0)

    def test_merged_samples_fail_the_identity_check_even_when_they_round_to_a_declared_id(self) -> None:
        merged = [
            [(0, 1, 0, 0.5, 5.0, 2.0, 0.5, 1, 0.5, 3), (0, 1, 0, 1.0, 5.0, 1.0, 1.0, 1)],
            [(0, 1, 0, 0.25, 9.0, 30000.4, 0.6, 30000, 0.4, 30001), (0, 1, 0, 1.0, 9.0, 1176.0, 1.0, 1176)],
            [(0, 1, 0, 1.0, 9.0, 18153.0, 1.0, 18153)],
        ]
        deep = _merged_deep(merged)
        self.assertEqual(
            instance_ids(deep).tolist(), [identity.UNLABELLED, 1, identity.UNLABELLED, 1176, 18153]
        )
        self.assertEqual(
            np.rint(deep.channels["instance_id"] / deep.channels["coverage"]).tolist()[:3], [2, 1, 30000]
        )
        beauty = {c: flatten(deep).rgba[..., i].astype(np.float32) for i, c in enumerate("RGB")}
        metrics = checks.deep_metrics(deep, beauty, [(0, 4), (1000, 32768)])
        self.assertEqual(metrics["valid_id_fraction"], 1.0)
        self.assertEqual(metrics["id_check_failure_fraction"], 0.4)
        self.assertAlmostEqual(metrics["unlabelled_coverage"], (0.5 + 0.25) / 3, places=6)
        self.assertGreater(metrics["unlabelled_coverage"], checks.LIMITS["unlabelled_coverage"][1])


def _merged_deep(pixels: list[list[tuple[float, ...]]]) -> Deep:
    """Pixels of (r, g, b, a, z, id quotient, share, id, ...) samples; checks and roles mix the sources."""
    flat = [s for p in pixels for s in p]
    a = np.array([s[3] for s in flat], np.float32)
    channels = {name: np.array([s[i] for s in flat], np.float32) * a for i, name in enumerate("RGB")}
    channels |= _labels([[(float(s[i]), int(s[i + 1])) for i in range(6, len(s), 2)] for s in flat], a)
    channels["Z"] = np.array([s[4] for s in flat], np.float32)
    channels["ZBack"] = channels["Z"].copy()
    channels["instance_id"] = np.array([s[5] for s in flat], np.float32) * a
    channels["coverage"] = a.copy()
    channels["A"] = a.astype(np.float16).astype(np.float32)
    return Deep(len(pixels), 1, np.array([len(p) for p in pixels]), channels)


class RoleChannelTest(unittest.TestCase):
    def test_planar_role_gather_matches_prefix_minimum(self) -> None:
        alpha = np.array([0.3, 0.5, 1.0], np.float32)
        order = np.array([2, 0, 1], np.int64)
        for dtype in (np.float16, np.float32, np.float64):
            raw = np.array([[0.8, 0.9, 0.2], [0.2, 0.1, -0.1], [0.7, 0.6, 0.4]], dtype)
            expected = np.minimum.accumulate(np.maximum(np.column_stack([alpha, raw[order]]), 0), axis=1)
            np.testing.assert_array_equal(
                roles.nested(tuple(raw[:, i] for i in range(3)), alpha, order=order),
                expected[:, 1:].astype(np.float32),
            )
            np.testing.assert_array_equal(roles.nested(raw[order], alpha), expected[:, 1:].astype(np.float32))
        invalid = (np.zeros((3, 2)), (raw[:, 0], raw[:, 1]), (raw[:, 0], raw[:2, 1], raw[:, 2]))
        for channels in invalid:
            with self.assertRaises(ValueError):
                roles.nested(channels, alpha)
        with self.assertRaises(ValueError):
            roles.nested(raw, alpha[:2])
        with self.assertRaises(ValueError):
            roles.nested(raw, alpha, order=np.array([0, 1, 3], np.int64))

    def test_role_channels_out_of_order_fail_raw_qa(self) -> None:
        deep = _deep([[(0, 0, 1, 1.0, 5.0, 3)], [(0, 1, 0, 0.5, 2.0, 2), (0, 1, 0, 1.0, 3000.0, 0)]])
        beauty = {c: flatten(deep).rgba[..., i].astype(np.float32) for i, c in enumerate("RGB")}
        ranges = [(0, 4)]
        metrics = checks.deep_metrics(deep, beauty, ranges)
        self.assertEqual(metrics["role_excess"], 0.0)
        for name, low, high, excess in (("fur", 2, 3, 0.25), ("foreground", 0, 1, 0.5), ("hero", 0, 1, 1.0)):
            broken = {**deep.channels, name: deep.channels[name].copy()}
            broken[name][low:high] += excess
            found = checks.deep_metrics(Deep(deep.width, deep.height, deep.counts, broken), beauty, ranges)
            self.assertAlmostEqual(found["role_excess"], excess, places=6)
            self.assertGreater(found["role_excess"], checks.LIMITS["role_excess"][1])

    def test_pack_refuses_missing_identity_or_role_channels_without_publishing(self) -> None:
        for missing in (
            "coverage",
            "instance_id",
            "instance_check",
            "instance_check2",
            "foreground",
            "hero",
            "fur",
        ):
            with self.subTest(channel=missing), tempfile.TemporaryDirectory() as tmp:
                shot = Path(tmp) / "shot"
                write_shot(shot, [[Sample(0.2, 1.0, 2)]])
                raw = shot / "deep" / "0001.exr"
                source = OpenEXR.File(str(raw), separate_channels=True).parts[0]
                channels = {name: value for name, value in source.channels.items() if name != missing}
                OpenEXR.File(source.header, channels).write(str(raw))
                before = raw.read_bytes()
                plan = pack.ShotFrames.load(shot)
                with self.assertRaises(KeyError) as raised:
                    pack.pack(shot, range(1, 2))
                self.assertEqual(raised.exception.args, (missing,))
                self.assertEqual(raw.read_bytes(), before)
                self.assertFalse(plan.merged_dir(shot, 1).exists())
                self.assertFalse(plan.sample_dir(shot, 1).exists())
                self.assertFalse((shot / "layers" / "0001.exr").exists())


class IdentityTest(unittest.TestCase):
    def test_check_is_a_pinned_hash_in_one_to_two(self) -> None:
        ids = np.arange(identity.MAX_ID)
        values = identity.check(ids)
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.all((values >= 1.0) & (values < 2.0)))
        np.testing.assert_array_equal(values, identity.check(ids.copy()))
        self.assertEqual(identity.check(np.array([0, 1, 2, 3, 1000])).tolist(), PINNED_CHECKS)
        for outside in (-1, identity.MAX_ID):
            with self.assertRaises(ValueError):
                identity.check(np.array([outside]))

    def test_neighbouring_ids_get_unrelated_checks(self) -> None:
        values = identity.check(np.arange(identity.MAX_ID)).astype(np.float64)
        steps = np.abs(np.diff(values))
        self.assertLess(float((steps <= 2 * identity.TOLERANCE).mean()), 8 * identity.TOLERANCE)
        self.assertAlmostEqual(float(steps.mean()), 1 / 3, delta=0.01)
        bits = identity.check(np.arange(identity.MAX_ID)).view(np.uint32)
        flipped = np.unpackbits((bits[1:] ^ bits[:-1]).view(np.uint8)).reshape(-1, 32).sum(axis=1)
        self.assertAlmostEqual(float(flipped.mean()), 23 / 2, delta=0.1)

    def test_the_second_check_is_pinned_and_unrelated_to_the_first(self) -> None:
        ids = np.arange(identity.MAX_ID)
        first, second = identity.check(ids).astype(np.float64), identity.check2(ids).astype(np.float64)
        self.assertTrue(np.all((second >= 1.0) & (second < 2.0)))
        self.assertEqual(identity.check2(np.array([0, 1, 2, 3, 1000])).tolist(), PINNED_CHECKS2)
        self.assertLess(abs(float(np.corrcoef(first, second)[0, 1])), 0.02)
        self.assertLess(abs(float(np.corrcoef(np.diff(first), np.diff(second))[0, 1])), 0.02)

    def test_the_checks_of_nearby_ids_differ_independently(self) -> None:
        ids = np.arange(identity.MAX_ID)
        first, second = identity.check(ids).astype(np.float64), identity.check2(ids).astype(np.float64)
        for close in (2.0**-5, 2.0**-7):
            near = np.zeros((2, 2), np.int64)
            for distance in range(1, 9):
                low, high = ids[:-distance], ids[distance:]
                a = np.abs(first[high] - first[low]) <= close
                b = np.abs(second[high] - second[low]) <= close
                near += np.array([[a.sum(), b.sum()], [(a & b).sum(), low.size]])
            expected = near[0, 0] * near[0, 1] / near[1, 1]
            self.assertLess(abs(near[1, 0] - expected), 4.0 * np.sqrt(expected), close)

    def test_two_source_merges_of_nearby_ids_pass_within_the_stated_bound(self) -> None:
        for share in (1 / 2, 1 / 4, 1 / 16, 1 / 64, 1 / 256):
            accepted, total = 0, 0
            for distance in range(1, 9):
                low = np.arange(identity.MAX_ID - distance)
                for kept, other in ((low, low + distance), (low + distance, low)):
                    decoded = _decode_merge([(kept, 1.0 - share), (other, share)])
                    self.assertTrue(np.all((decoded == kept) | (decoded == identity.UNLABELLED)))
                    accepted += int((decoded == kept).sum())
                    total += kept.size
            expected = _bound(share) * total
            self.assertLessEqual(accepted, expected + 4.0 * np.sqrt(expected) + 1.0, share)

    def test_many_source_merges_pass_within_the_stated_bound(self) -> None:
        rng = np.random.default_rng(11)
        count = 200_000
        for share in (1 / 2, 1 / 10, 1 / 100):
            accepted = 0
            for others in range(1, 5):
                kept = rng.integers(0, identity.MAX_ID, count)
                steps = rng.integers(1, 9, (others, count)) * rng.choice([-1, 1], (others, count))
                weights = rng.dirichlet(np.ones(others), count).T * share
                sources = [(kept, np.full(count, 1.0 - share))]
                sources += [((kept + steps[k]) % identity.MAX_ID, weights[k]) for k in range(others)]
                accepted += int((_decode_merge(sources) != identity.UNLABELLED).sum())
            expected = _bound(share) * 4 * count
            self.assertLessEqual(accepted, expected + 4.0 * np.sqrt(expected) + 1.0, share)
        absent = _decode_merge([(np.full(count, 1), 0.5), (np.full(count, 3), 0.5)])
        self.assertTrue(np.all(absent == identity.UNLABELLED))

    def test_merges_of_the_fixed_ids_never_keep_a_mislabelled_id(self) -> None:
        grid = np.array([c for c in np.ndindex(*(33,) * 4) if sum(c) <= 32], np.float64)
        shares = np.concatenate([grid, 32 - grid.sum(axis=1, keepdims=True)], axis=1) / 32
        decoded = _decode_merge([(np.full(len(shares), k), shares[:, k]) for k in range(5)])
        kept = decoded != identity.UNLABELLED
        self.assertTrue(np.all(shares[kept, decoded[kept]] == 1.0))
        self.assertEqual(int(kept.sum()), 5)

    def test_reviewed_merges_lose_their_id(self) -> None:
        cases = [
            [(1, 107 / 256), (3, 113 / 256), (4, 36 / 256)],
            [(1, 0.5), (3, 0.5)],
            [(1, 0.5), (2001, 0.5)],
            [(30000, 0.6), (30001, 0.4)],
            [(13456, 0.51), (13457, 0.49)],
            [(99, 0.99), (100, 0.01)],
            [(2, 0.99), (3, 0.01)],
        ]
        for case in cases:
            sources = [(np.array([i]), w) for i, w in case]
            self.assertEqual(_decode_merge(sources).tolist(), [identity.UNLABELLED], case)

    def test_single_check_collisions_of_adjacent_ids_fail_the_second_check(self) -> None:
        low = np.arange(identity.MAX_ID - 1)
        first = identity.check(low).astype(np.float64), identity.check(low + 1).astype(np.float64)
        collided = low[np.abs(0.01 * (first[1] - first[0])) <= identity.TOLERANCE]
        self.assertGreater(collided.size, 300)
        decoded = _decode_merge([(collided, 0.99), (collided + 1, 0.01)])
        self.assertLess(int((decoded != identity.UNLABELLED).sum()), collided.size // 20)

    def test_a_dominant_source_keeps_its_id_when_the_rest_is_below_the_tolerance(self) -> None:
        sources = [(np.array([2]), 1.0 - 2.0**-17), (np.array([3]), 2.0**-17)]
        self.assertEqual(_decode_merge(sources).tolist(), [2])

    def test_a_quotient_off_its_integer_loses_the_id_even_when_both_checks_match(self) -> None:
        ids = np.array([1, 1, 2, 18000, 18000])
        offsets = np.array([0.04, 0.3, -0.2, 0.3, 0.4])
        coverage = np.full(ids.size, 0.5, np.float32)
        decoded = identity.decode(
            ((ids + offsets) * coverage).astype(np.float32),
            identity.check(ids) * coverage,
            identity.check2(ids) * coverage,
            coverage,
        )
        self.assertEqual(
            decoded.tolist(), [1, identity.UNLABELLED, identity.UNLABELLED, 18000, identity.UNLABELLED]
        )

    def test_unmerged_ids_survive_float32_premultiplication(self) -> None:
        ids = np.arange(identity.MAX_ID)
        coverage = np.random.default_rng(3).uniform(1 / 64, 1.0, ids.size).astype(np.float32)
        decoded = identity.decode(
            ids.astype(np.float32) * coverage,
            identity.check(ids) * coverage,
            identity.check2(ids) * coverage,
            coverage,
        )
        np.testing.assert_array_equal(decoded, ids)
        order = np.array([identity.MAX_ID - 1, 0, 100, 0], np.int64)
        channels = (
            ids.astype(np.float32) * coverage,
            identity.check(ids) * coverage,
            identity.check2(ids) * coverage,
            coverage,
        )
        np.testing.assert_array_equal(identity.decode(*channels, order=order), ids[order])
        for outside in (-1, identity.MAX_ID):
            with self.assertRaises(ValueError):
                identity.decode(*channels, order=np.array([outside], np.int64))
        for dtype in (np.float16, np.float32, np.float64):
            zero = np.zeros(1, dtype)
            self.assertEqual(identity.decode(zero, zero, zero, zero).tolist(), [identity.UNLABELLED])
        self.assertEqual(identity.range_errors(0, identity.MAX_ID), [])
        self.assertTrue(identity.range_errors(identity.MAX_ID - 1, 2))


def _bound(share: float) -> float:
    """The documented chance that a merge whose mislabelled share is `share` keeps an id."""
    return (2.0 * identity.TOLERANCE / share + 2.0**-23) ** 2


def _decode_merge(
    sources: "Sequence[tuple[NDArray[np.int64], NDArray[np.float64] | float]]",
) -> "NDArray[np.int64]":
    """Decode samples Karma merged from (ids, shares) sources, stored at coverage 0.75 in float32."""
    count = max(ids.size for ids, _ in sources)
    sums = np.zeros((3, count))
    for ids, share in sources:
        every = np.broadcast_to(ids, (count,))
        weights = np.broadcast_to(np.asarray(share, np.float64), (count,))
        sums += weights * np.stack(
            [
                every.astype(np.float64),
                *(h(every).astype(np.float64) for h in (identity.check, identity.check2)),
            ]
        )
    stored = (sums * 0.75).astype(np.float32)
    return identity.decode(stored[0], stored[1], stored[2], np.full(count, 0.75, np.float32))


PINNED_CHECKS = [
    1.5733975172042847,
    1.5883936882019043,
    1.7041336297988892,
    1.9277628660202026,
    1.1540075540542603,
]
PINNED_CHECKS2 = [
    1.99704110622406,
    1.52652907371521,
    1.0283201932907104,
    1.6577625274658203,
    1.535722255706787,
]


class GroomTest(unittest.TestCase):
    def setUp(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float64)
        st = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], np.float64)
        self.skin = groom.Skin(points, np.array([[0, 1, 2], [0, 2, 3]]), st)
        self.params = groom.GroomParams(
            count=2000,
            segments=3,
            width_root_m=0.001,
            width_tip_m=0.0001,
            length_m=0.05,
            length_jitter=0.0,
            frizz_m=0.0,
            neighbours=1,
        )

    def test_hairs_follow_nearby_guides_and_skip_bare_skin(self) -> None:
        roots = np.array(
            [[x, 0.0, z] for x in np.linspace(0.05, 0.45, 5) for z in np.linspace(0.05, 0.95, 10)]
        )
        shape = np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.01, 0.04, 0.0], [0.02, 0.06, 0.0]])
        guides = groom.Guides(roots[:, None] + shape[None])
        hair = groom.build(self.skin, guides, self.params, stream(1, "g"))
        self.assertTrue(0 < len(hair.curves) < self.params.count)
        self.assertLess(float(hair.curves[:, 0, 0].max()), 0.45 + 4 * 0.1)
        np.testing.assert_allclose(
            hair.curves - hair.curves[:, :1], np.broadcast_to(shape, hair.curves.shape), atol=1e-9
        )

    def test_parametric_hair_grows_along_the_normal(self) -> None:
        hair = groom.build(self.skin, None, self.params, stream(1, "g"))
        self.assertEqual(len(hair.curves), self.params.count)
        tips = hair.curves[:, -1] - hair.curves[:, 0]
        self.assertTrue((np.abs(tips[:, 1]) > 0.02).all())
        self.assertEqual(hair.widths.shape, hair.curves.shape[:2])


class HdriTest(unittest.TestCase):
    def test_sun_cut_recovers_direction_and_energy(self) -> None:
        image = np.full((256, 512, 3), 0.5, np.float32)
        image[60, 300] = 4000.0
        clamped, cut = hdri.cut_sun(image)
        dirs, solid = hdri.directions(256, 512)
        np.testing.assert_allclose(cut.direction, dirs[60, 300], atol=1e-6)
        self.assertAlmostEqual(cut.irradiance, (4000.0 - 0.5) * float(solid[60, 0]), delta=1e-3)
        self.assertLess(float(clamped.max()), 1.0)


class RelativeAssetTest(unittest.TestCase):
    def test_paths_through_a_symlinked_assets_dir_stay_inside_the_build_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside, root = Path(tmp) / "store" / "hdri", Path(tmp) / "root"
            outside.mkdir(parents=True)
            (outside / "sky.exr").write_bytes(b"")
            (root / "assets").mkdir(parents=True)
            (root / "assets" / "hdri").symlink_to(outside, target_is_directory=True)
            scene = root / "shots" / "s" / "scene"
            scene.mkdir(parents=True)
            path = relative_asset(root / "assets" / "hdri" / "sky.exr", scene)
            self.assertEqual(path, "../../../assets/hdri/sky.exr")
            moved = Path(tmp) / "moved"
            root.rename(moved)
            self.assertTrue((moved / "shots" / "s" / "scene" / path).is_file())


class SettingsTest(unittest.TestCase):
    def test_deep_and_beauty_share_a_box_filter_and_per_frame_files(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        config = settings.RenderConfig(64, 36, 8, (1, 2), motion_blur=False, depth_of_field=True)
        render = settings.author(stage, "/cam", config)
        prim = render.GetPrim()
        self.assertEqual(prim.GetAttribute("karma:global:pixelfilter").Get(), "box")
        self.assertTrue(prim.GetAttribute("karma:global:abortnogpudevice").Get())
        deep = stage.GetPrimAtPath("/Render/products/deep")
        self.assertEqual(
            [p.name for p in deep.GetRelationship("orderedVars").GetTargets()], list(settings.DEEP_AOVS)
        )
        self.assertEqual(deep.GetAttribute("productName").Get(2), "deep/0002.exr")
        filters = json.loads(prim.GetAttribute("karma:global:imagefilter").Get())
        self.assertEqual(filters[0][1]["aovs"], ["denoised"])
        names = {aov.name: aov for aov in settings.BEAUTY_AOVS}
        self.assertEqual(names["denoised"].source_name, names["C"].source_name)
        self.assertTrue({"albedo", "N"} <= names.keys())
        self.assertEqual((names["foreground"].file_format, names["coverage"].file_format), ("float", "float"))
        for aov in settings.BEAUTY_AOVS:
            var = stage.GetPrimAtPath(f"/Render/vars/{aov.name}")
            self.assertIn(
                var.GetAttribute("driver:parameters:aov:karma:filter").Get(),
                ('["ubox",{}]', '["minmax",{"mode":"zmin"}]'),
            )

    def test_every_tagged_id_carries_its_check_into_the_deep(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        one = UsdGeom.Mesh.Define(stage, "/one").GetPrim()
        many = UsdGeom.PointInstancer.Define(stage, "/many").GetPrim()
        sky = UsdGeom.Mesh.Define(stage, "/sky").GetPrim()
        tag_instances(one, np.array([3]), "hero_groom")
        tag_instances(many, np.array([1000, 1001, identity.MAX_ID - 1]), "rock")
        tag_instances(sky, np.array([0]), "sky")
        for name, hashed in (("instance_check", identity.check), ("instance_check2", identity.check2)):
            found = UsdGeom.PrimvarsAPI(one).GetPrimvar(name)
            self.assertEqual(found.GetInterpolation(), UsdGeom.Tokens.constant)
            self.assertEqual(np.float32(found.Get()), hashed(np.array([3]))[0])
            values = np.asarray(UsdGeom.PrimvarsAPI(many).GetPrimvar(name).Get(), np.float32)
            np.testing.assert_array_equal(values, hashed(np.array([1000, 1001, identity.MAX_ID - 1])))
        for prim, expected in ((one, (1.0, 1.0, 1.0)), (sky, (0.0, 0.0, 0.0))):
            found = tuple(float(UsdGeom.PrimvarsAPI(prim).GetPrimvar(r).Get()) for r in roles.ROLES)
            self.assertEqual(found, expected)
        for role, expected in zip(roles.ROLES, (1.0, 0.0, 0.0), strict=True):
            values = np.asarray(UsdGeom.PrimvarsAPI(many).GetPrimvar(role).Get(), np.float32)
            self.assertEqual(values.tolist(), [expected] * 3)
        with self.assertRaises(ValueError):
            tag_instances(
                UsdGeom.Mesh.Define(stage, "/outside").GetPrim(), np.array([identity.MAX_ID]), "rock"
            )
        settings.author(
            stage, "/cam", settings.RenderConfig(4, 4, 8, (1,), motion_blur=False, depth_of_field=False)
        )
        self.assertTrue({"instance_check", "instance_check2", *roles.ROLES} <= set(settings.DEEP_AOVS))
        names = {aov.name: aov for aov in settings.BEAUTY_AOVS}
        self.assertEqual(
            (names["instance_check"].source_type, names["instance_check"].file_format), ("primvar", "float")
        )

    def test_deep_never_merges_samples_across_depths(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        settings.author(
            stage, "/cam", settings.RenderConfig(64, 36, 8, (1,), motion_blur=False, depth_of_field=False)
        )
        deep = stage.GetPrimAtPath("/Render/products/deep")
        self.assertEqual(deep.GetAttribute("karma:product:dcmzbias").Get(), 0.0)
        self.assertEqual(deep.GetAttribute("karma:product:dcmcompression").Get(), 5)


if __name__ == "__main__":
    unittest.main()
