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
from strata_datagen.qa.deep import flatten, instance_ids
from strata_datagen.render import settings
from strata_datagen.rng import stream
from strata_datagen.usdutil import relative_asset, tag_instances
from strata_dataset import identity

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    channels["instance_check"] = identity.check(np.rint(ids).astype(np.int64)) * a
    channels["coverage"] = a.copy()
    channels["A"] = a.astype(np.float16).astype(np.float32)
    return Deep(len(samples), 1, counts, channels)


class DeepTest(unittest.TestCase):
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
    checks_cells = np.empty((1, len(samples)), dtype=object)
    for i, pixel in enumerate(samples):
        alpha = np.array([s[1] for s in pixel], np.float32)
        checks_cells[0, i] = identity.check(np.array([int(s[2]) for s in pixel], np.int64)) * alpha
    chans["instance_check"] = checks_cells
    for name in ("Z", "ZBack"):
        cells = np.empty((1, len(samples)), dtype=object)
        for i, pixel in enumerate(samples):
            cells[0, i] = np.array([s[3] if len(s) > 3 else 2.0 for s in pixel], np.float32)
        chans[name] = cells
    header: dict[str, object] = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.deepscanline}
    OpenEXR.File(header, {k: OpenEXR.Channel(v) for k, v in chans.items()}).write(str(path))


class QaTest(unittest.TestCase):
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
            self.assertGreater(checks.deep_metrics(read_deep(path), beauty, ranges)["sky_id_in_front"], 0.0)

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
        self.assertGreater(
            metrics["id_check_failure_fraction"], checks.LIMITS["id_check_failure_fraction"][1]
        )


def _merged_deep(pixels: list[list[tuple[float, ...]]]) -> Deep:
    """Pixels of (r, g, b, a, z, id quotient, share, id, ...) samples; the check averages its sources."""
    flat = [s for p in pixels for s in p]
    a = np.array([s[3] for s in flat], np.float32)
    checks = [
        sum(float(s[i]) * float(identity.check(np.array([int(s[i + 1])]))[0]) for i in range(6, len(s), 2))
        for s in flat
    ]
    channels = {name: np.array([s[i] for s in flat], np.float32) * a for i, name in enumerate("RGB")}
    channels["Z"] = np.array([s[4] for s in flat], np.float32)
    channels["ZBack"] = channels["Z"].copy()
    channels["instance_id"] = np.array([s[5] for s in flat], np.float32) * a
    channels["instance_check"] = np.array(checks, np.float32) * a
    channels["coverage"] = a.copy()
    channels["A"] = a.astype(np.float16).astype(np.float32)
    return Deep(len(pixels), 1, np.array([len(p) for p in pixels]), channels)


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

    def test_merges_of_different_ids_pass_the_check_at_the_stated_rate(self) -> None:
        rng = np.random.default_rng(7)
        count = 400_000
        first = rng.integers(0, identity.MAX_ID, count)
        second = (first + rng.integers(1, identity.MAX_ID, count)) % identity.MAX_ID
        share = rng.choice([0.5, 0.25, 1 / 3, 1 / 64], count)
        quotient = share * first + (1 - share) * second
        check = share * identity.check(first) + (1 - share) * identity.check(second)
        coverage = rng.uniform(1 / 64, 1.0, count)
        decoded = identity.decode(
            (quotient * coverage).astype(np.float32),
            (check * coverage).astype(np.float32),
            coverage.astype(np.float32),
        )
        self.assertLess(float((decoded != identity.UNLABELLED).mean()), 4 * identity.TOLERANCE)

    def test_unmerged_ids_survive_float32_premultiplication(self) -> None:
        ids = np.arange(identity.MAX_ID)
        coverage = np.random.default_rng(3).uniform(1 / 64, 1.0, ids.size).astype(np.float32)
        decoded = identity.decode(ids.astype(np.float32) * coverage, identity.check(ids) * coverage, coverage)
        np.testing.assert_array_equal(decoded, ids)
        zero = np.zeros(1, np.float32)
        self.assertEqual(identity.decode(zero, zero, zero).tolist(), [identity.UNLABELLED])
        self.assertEqual(identity.range_errors(0, identity.MAX_ID), [])
        self.assertTrue(identity.range_errors(identity.MAX_ID - 1, 2))


PINNED_CHECKS = [
    1.5733975172042847,
    1.5883936882019043,
    1.7041336297988892,
    1.9277628660202026,
    1.1540075540542603,
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
        tag_instances(one, np.array([3]))
        tag_instances(many, np.array([1000, 1001, identity.MAX_ID - 1]))
        found = UsdGeom.PrimvarsAPI(one).GetPrimvar("instance_check")
        self.assertEqual(found.GetInterpolation(), UsdGeom.Tokens.constant)
        self.assertEqual(np.float32(found.Get()), identity.check(np.array([3]))[0])
        values = np.asarray(UsdGeom.PrimvarsAPI(many).GetPrimvar("instance_check").Get(), np.float32)
        np.testing.assert_array_equal(values, identity.check(np.array([1000, 1001, identity.MAX_ID - 1])))
        with self.assertRaises(ValueError):
            tag_instances(UsdGeom.Mesh.Define(stage, "/outside").GetPrim(), np.array([identity.MAX_ID]))
        settings.author(
            stage, "/cam", settings.RenderConfig(4, 4, 8, (1,), motion_blur=False, depth_of_field=False)
        )
        self.assertIn("instance_check", settings.DEEP_AOVS)
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
