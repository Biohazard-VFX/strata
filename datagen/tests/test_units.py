import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import OpenEXR
from pxr import Usd

from strata_datagen.exr import Deep
from strata_datagen.generators import groom
from strata_datagen.library import hdri
from strata_datagen.qa import checks
from strata_datagen.qa.deep import flatten, instance_ids
from strata_datagen.render import settings
from strata_datagen.rng import stream

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _deep(samples: list[list[tuple[float, float, float, float, float, int]]]) -> Deep:
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
    channels["instance_id"] = np.array([s[5] for s in flat], np.float32) * a
    return Deep(len(samples), 1, counts, channels)


class DeepTest(unittest.TestCase):
    def test_flatten_composites_front_to_back_and_flags_see_through_pixels(self) -> None:
        deep = _deep(
            [
                [(0, 0, 1, 1.0, 5.0, 1000), (1, 0, 0, 0.5, 1.0, 7)],
                [(0, 1, 0, 1.0, 2.0, 2)],
                [(0, 1, 0, 0.3, 2.0, 3)],
            ]
        )
        flat = flatten(deep)
        np.testing.assert_allclose(flat.rgba[0, 0], [0.5, 0.0, 0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(flat.rgba[0, 2], [0.0, 0.3, 0.0, 0.3], atol=1e-6)
        self.assertEqual(flat.partial[0].tolist(), [True, False, True])
        self.assertEqual(instance_ids(deep).tolist(), [1000, 7, 2, 3])


def _write_deep(path: Path, samples: list[list[tuple[float, float, float]]]) -> None:
    """Pixels of (premultiplied grey, alpha, id) samples at depth 2 m in a 2x1 deep image."""
    chans: dict[str, NDArray[np.object_]] = {}
    for name, index in (("R", 0), ("G", 0), ("B", 0), ("A", 1), ("instance_id", 2)):
        cells = np.empty((1, len(samples)), dtype=object)
        for i, pixel in enumerate(samples):
            value = np.array([s[index] * (s[1] if name == "instance_id" else 1.0) for s in pixel], np.float32)
            cells[0, i] = value
        chans[name] = cells
    for name in ("Z", "ZBack"):
        cells = np.empty((1, len(samples)), dtype=object)
        for i, pixel in enumerate(samples):
            cells[0, i] = np.full(len(pixel), 2.0, np.float32)
        chans[name] = cells
    header: dict[str, object] = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.deepscanline}
    OpenEXR.File(header, {k: OpenEXR.Channel(v) for k, v in chans.items()}).write(str(path))


class QaTest(unittest.TestCase):
    def test_sky_behind_partial_pixels_is_allowed_but_deep_may_not_exceed_beauty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.exr"
            _write_deep(path, [[(0.5, 1.0, 1000)], [(0.2, 0.5, 2)]])
            sky = np.array([[0.5, 0.2 + 0.5 * 0.6]], np.float32)
            beauty = dict.fromkeys("RGB", sky)
            metrics = checks.deep_metrics(path, beauty, [(1, 4), (1000, 1001)])
            self.assertLess(metrics["flatten_error"], 1e-6)
            self.assertLess(metrics["flatten_excess"], 1e-6)
            self.assertEqual(metrics["valid_id_fraction"], 1.0)
            dark = {c: np.array([[0.5, 0.1]], np.float32) for c in "RGB"}
            self.assertGreater(
                checks.deep_metrics(path, dark, [(1, 4), (1000, 1001)])["flatten_excess"], 0.04
            )


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
        for aov in settings.BEAUTY_AOVS:
            var = stage.GetPrimAtPath(f"/Render/vars/{aov.name}")
            self.assertIn(
                var.GetAttribute("driver:parameters:aov:karma:filter").Get(),
                ('["ubox",{}]', '["minmax",{"mode":"zmin"}]'),
            )


if __name__ == "__main__":
    unittest.main()
