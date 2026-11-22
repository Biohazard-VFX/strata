import filecmp
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from pxr import Sdf, Usd, UsdGeom
from scipy.spatial import KDTree

from strata_datagen import timeline
from strata_datagen.generators import hairstyle
from strata_datagen.library import heroes
from strata_datagen.library.catalogue import Hero, HeroHair, HeroPose, Library, Provenance
from strata_datagen.rng import stream
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import ShotSpec
from strata_datagen.usdutil import FloatArray, IntArray, floats, ints, new_stage, vec2f, vec3f
from tests.test_determinism import RENDER, library

CENTRE = np.array([0.0, 1.6, 0.0])
RADIUS = 0.09


def sphere(rings: int = 24, segments: int = 32) -> tuple[FloatArray, IntArray]:
    theta = np.linspace(0.0, np.pi, rings + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    points = np.stack([np.sin(t) * np.sin(p), np.cos(t), np.sin(t) * np.cos(p)], -1).reshape(-1, 3)
    quads = [
        (
            r * segments + s,
            (r + 1) * segments + s,
            (r + 1) * segments + (s + 1) % segments,
            r * segments + (s + 1) % segments,
        )
        for r in range(rings)
        for s in range(segments)
    ]
    triangles = np.array([tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))], dtype=np.int64)
    return CENTRE + RADIUS * points, triangles


def regions(points: FloatArray) -> dict[str, FloatArray]:
    q = (points - CENTRE) / RADIUS
    face = q[:, 2] > 0.55
    return {
        "scalp": ((q[:, 1] > -0.3) & ~(face & (q[:, 1] < 0.45))).astype(np.float64),
        "brows": (face & (q[:, 1] > 0.2) & (q[:, 1] < 0.35) & (np.abs(q[:, 0]) > 0.1)).astype(np.float64),
        "lashes": (face & (q[:, 1] > 0.08) & (q[:, 1] < 0.16) & (np.abs(q[:, 0]) > 0.15)).astype(np.float64),
        "beard": (face & (q[:, 1] < -0.2)).astype(np.float64) + ((q[:, 2] > 0.2) & (q[:, 1] < -0.45)),
    }


def head() -> hairstyle.Head:
    points, triangles = sphere()
    marks = {k: np.clip(v, 0.0, 1.0) for k, v in regions(points).items()}
    return hairstyle.Head(points, triangles, marks, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def style(seed: int, **changes: object) -> hairstyle.Style:
    drawn = hairstyle.sample(stream(seed, "hairstyle"), hairstyle.REGIONS, facial_hair=True)
    return replace(drawn, **{"density_per_cm2": 3.0, **changes})


def build(drawn: hairstyle.Style) -> hairstyle.Hairstyle:
    return hairstyle.build(head(), drawn, stream(0, "hairstyle", "build"))


def scalp(built: hairstyle.Hairstyle) -> hairstyle.StrandSet:
    return first_key(next(s for s in built.sets if s.name == "scalp"))


def first_key(strand_set: hairstyle.StrandSet) -> hairstyle.StrandSet:
    return hairstyle.StrandSet(strand_set.name, strand_set.curves[0], strand_set.widths[0])


class SampleTest(unittest.TestCase):
    def test_seeds_cover_every_style_choice(self) -> None:
        drawn = [
            hairstyle.sample(stream(s, "hairstyle"), hairstyle.REGIONS, facial_hair=True) for s in range(400)
        ]
        self.assertEqual({d.cut for d in drawn}, set(hairstyle.CUTS))
        self.assertEqual({d.texture for d in drawn}, set(hairstyle.TEXTURES))
        self.assertEqual({d.updo for d in drawn}, set(hairstyle.UPDOS))
        self.assertEqual({d.facial for d in drawn}, set(hairstyle.FACIAL))
        self.assertEqual({d.colour.name for d in drawn}, {c[0] for c in hairstyle.COLOURS})
        self.assertTrue(any(d.wet for d in drawn) and any(d.bangs for d in drawn))
        self.assertTrue(all(d.updo == "down" for d in drawn if d.cut in ("buzz", "short")))

    def test_head_limits_what_is_drawn(self) -> None:
        for seed in range(60):
            bare = hairstyle.sample(stream(seed, "hairstyle"), ("scalp",), facial_hair=True)
            barred = hairstyle.sample(stream(seed, "hairstyle"), hairstyle.REGIONS, facial_hair=False)
            full = hairstyle.sample(stream(seed, "hairstyle"), hairstyle.REGIONS, facial_hair=True)
            self.assertEqual((bare.facial, bare.brows, bare.lashes), ("none", False, False))
            self.assertEqual(barred.facial, "none")
            completed = replace(bare, facial=full.facial, brows=True, lashes=True)
            self.assertEqual(hairstyle.record(completed), hairstyle.record(full))


class BuildTest(unittest.TestCase):
    def test_same_seed_gives_identical_strands(self) -> None:
        first, second = build(style(3)), build(style(3))
        self.assertEqual([s.name for s in first.sets], [s.name for s in second.sets])
        for a, b in zip(first.sets, second.sets, strict=True):
            np.testing.assert_array_equal(a.curves, b.curves)
            np.testing.assert_array_equal(a.widths, b.widths)

    def test_strands_root_on_their_regions_and_stay_outside_the_skin(self) -> None:
        marks = regions(head().points)
        for seed in (2, 4, 17, 51):
            built = build(style(seed))
            for strand_set in map(first_key, built.sets):
                with self.subTest(seed=seed, region=strand_set.name):
                    radius = np.linalg.norm(strand_set.curves - CENTRE, axis=-1)
                    np.testing.assert_allclose(radius[:, 0], RADIUS, atol=0.004)
                    self.assertGreater(radius.min(), RADIUS * 0.97)
                    nearest = corners((strand_set.curves[:, 0] - CENTRE) / RADIUS)
                    self.assertTrue((marks[strand_set.name][nearest].max(axis=1) > 0.0).all())

    def test_widths_taper_to_the_tip(self) -> None:
        for strand_set in map(first_key, build(style(5)).sets):
            self.assertTrue((strand_set.widths > 0).all())
            self.assertTrue((strand_set.widths[:, -1] < strand_set.widths[:, 0]).all())

    def test_long_hair_hangs_below_short_hair(self) -> None:
        low = {
            cut: scalp(build(style(9, cut=cut, length_m=length, updo="down"))).curves[..., 1].min()
            for cut, length in (("short", 0.05), ("long", 0.4))
        }
        self.assertLess(low["long"], low["short"] - 0.2)

    def test_ponytail_gathers_at_one_tie(self) -> None:
        def tightest(updo: str) -> float:
            curves = scalp(build(style(17, updo=updo, length_m=0.4))).curves
            return float(np.linalg.norm(curves.std(axis=0), axis=-1).min())

        self.assertLess(tightest("ponytail"), 0.3 * tightest("down"))

    def test_curls_lengthen_the_path_of_the_same_cut(self) -> None:
        def path(texture: str, radius: float, turns: float) -> float:
            built = scalp(
                build(style(37, texture=texture, curl_radius_m=radius, curl_turns_per_m=turns, updo="down"))
            )
            return float(np.linalg.norm(np.diff(built.curves, axis=1), axis=-1).sum(axis=1).mean())

        self.assertGreater(path("curly", 0.008, 25.0), 1.2 * path("straight", 0.0, 0.0))

    def test_author_writes_one_curves_prim_per_set(self) -> None:
        built = build(style(52))
        stage = Usd.Stage.CreateInMemory()
        prims = hairstyle.author(stage, "/hair", built)
        for strand_set in map(first_key, built.sets):
            curves = prims[strand_set.name]
            counts = np.asarray(curves.GetCurveVertexCountsAttr().Get())
            self.assertEqual(counts.tolist(), [strand_set.curves.shape[1]] * len(strand_set.curves))
            self.assertEqual(len(curves.GetWidthsAttr().Get()), strand_set.widths.size)
            self.assertEqual(curves.GetPointsAttr().GetNumTimeSamples(), 0)


def drifting(seed: int, **changes: object) -> hairstyle.Style:
    """A style whose every drifting parameter swings well past its normal range over 30 s."""
    rng = stream(seed, "drift")
    curves = tuple(timeline.curve(rng, 30.0, 10.0, -1.6 * m, 1.6 * m) for m in hairstyle.DRIFT_MAX)
    return replace(style(seed, **changes), drift=hairstyle.Drift(curves, 0.0, 0.0))


class CostTest(unittest.TestCase):
    def test_point_cap_trades_strands_for_width_and_keeps_summed_width(self) -> None:
        drawn = replace(style(17, updo="down", cut="long", texture="straight"), drift=hairstyle.STILL)
        full = next(s for s in hairstyle.build(head(), drawn, stream(0, "c")).sets if s.name == "scalp")
        with mock.patch.object(hairstyle, "SHOT_POINTS", 3 * full.curves[0, ..., 0].size // 2):
            capped = next(
                s
                for s in hairstyle.build(head(), drawn, stream(0, "c"), (0.0, 6.0, 12.0)).sets
                if s.name == "scalp"
            )
        self.assertLess(capped.curves.shape[1], 0.55 * full.curves.shape[1])
        summed_full = float(full.widths[0, :, 0].sum())
        summed_capped = float(capped.widths[0, :, 0].sum())
        self.assertAlmostEqual(summed_capped / summed_full, 1.0, delta=0.1)


class MorphTest(unittest.TestCase):
    """A shot's hairstyle changes over time with one topology, so keys interpolate."""

    def test_keys_share_topology_and_roots(self) -> None:
        built = hairstyle.build(head(), drifting(37, updo="down"), stream(0, "b"), (0.0, 6.0, 12.0))
        for strand_set in built.sets:
            self.assertEqual(strand_set.curves.shape[0], 3)
            np.testing.assert_array_equal(strand_set.curves[0, :, 0], strand_set.curves[2, :, 0])

    def test_change_is_smooth_in_time_and_large_over_the_shot(self) -> None:
        built = hairstyle.build(head(), drifting(37, updo="down"), stream(0, "b"), (0.0, 1.0 / 24.0, 15.0))
        curves = scalp_keys(built)
        step = np.abs(curves[1] - curves[0]).max()
        span = np.abs(curves[2] - curves[0]).max()
        self.assertLess(step, 0.004)
        self.assertGreater(span, 10.0 * step)

    def test_length_follows_the_drift_and_density_fades_widths(self) -> None:
        drawn = drifting(17, updo="ponytail", length_m=0.4)
        times = tuple(float(t) for t in np.linspace(0.0, 30.0, 7))
        built = hairstyle.build(head(), drawn, stream(0, "b"), times)
        keys = next(s for s in built.sets if s.name == "scalp")
        lengths = np.linalg.norm(np.diff(keys.curves, axis=2), axis=-1).sum(axis=2).mean(axis=1)
        wanted = np.array([hairstyle.at(drawn, t).length_m for t in times])
        self.assertGreater(float(np.corrcoef(lengths, wanted)[0, 1]), 0.95)
        hidden = (keys.widths.max(axis=2) < 1e-9).mean(axis=1)
        density = np.array([hairstyle.at(drawn, t).density_per_cm2 for t in times])
        self.assertGreater(hidden.max(), 0.05)
        self.assertLess(float(np.corrcoef(hidden, density)[0, 1]), -0.9)

    def test_author_time_samples_every_key(self) -> None:
        built = hairstyle.build(head(), drifting(52), stream(0, "b"), (0.0, 3.0))
        stage = Usd.Stage.CreateInMemory()
        prims = hairstyle.author(stage, "/hair", built, (1.0, 73.0))
        for curves in prims.values():
            self.assertEqual(curves.GetPointsAttr().GetTimeSamples(), [1.0, 73.0])
            self.assertEqual(curves.GetWidthsAttr().GetTimeSamples(), [1.0, 73.0])


def scalp_keys(built: hairstyle.Hairstyle) -> FloatArray:
    return next(s for s in built.sets if s.name == "scalp").curves


def corners(q: FloatArray) -> IntArray:
    """Nearby skin vertices, enough to include the triangle corners even at the poles (32 coincide)."""
    _, index = KDTree((head().points - CENTRE) / RADIUS).query(q, k=40)
    return np.asarray(index, dtype=np.int64)


def write_hero(folder: Path) -> None:
    points, triangles = sphere()
    stage = new_stage(folder / "rest.usda")
    UsdGeom.Xform.Define(stage, "/body")
    mesh = UsdGeom.Mesh.Define(stage, "/body/skin")
    mesh.CreatePointsAttr(vec3f(points))
    mesh.CreateFaceVertexCountsAttr(ints(np.full(len(triangles), 3)))
    mesh.CreateFaceVertexIndicesAttr(ints(triangles.ravel()))
    primvars = UsdGeom.PrimvarsAPI(mesh)
    primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        vec2f(np.zeros((triangles.size, 2)))
    )
    for name, weight in regions(points).items():
        primvars.CreatePrimvar(f"region_{name}", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
            floats(np.clip(weight, 0.0, 1.0))
        )
    stage.GetRootLayer().Save()
    manifest = {
        "key": "sphere",
        "poses": {"rest": "rest.usda"},
        "skin_prim": "/body/skin",
        "albedo": "albedo.png",
        "regions": list(hairstyle.REGIONS),
        "facial_hair": True,
        "source": "test",
        "license": "test",
    }
    (folder / "hero.json").write_text(json.dumps(manifest))


def hair_library(root: Path) -> Library:
    source = root / "library" / "sphere"
    source.mkdir(parents=True)
    write_hero(source)
    manifest = heroes.load(source)
    prepared = heroes.prepare_pose(manifest, "rest", root / "assets" / "heroes" / "sphere" / "rest.usdc")
    pose = HeroPose("rest", "assets/heroes/sphere/rest.usdc", prepared.bounds_min, prepared.bounds_max, 0)
    hair = HeroHair(manifest.regions, manifest.forward, manifest.facial_hair)
    hero = Hero(
        "sphere", "assets/heroes/sphere/albedo.png", (pose,), Provenance("test", "test", "test"), hair
    )
    return replace(library(), heroes=(hero,))


class ComposeTest(unittest.TestCase):
    def test_shots_grow_a_seeded_hairstyle_on_the_hero(self) -> None:
        with TemporaryDirectory() as tmp:
            shots: list[Path] = []
            for name in ("a", "b"):
                root = Path(tmp) / name
                lib = hair_library(root)
                shot = compose.plan(ShotSpec(seed=11), lib, root)
                compose.write(shot, lib, root, RENDER)
                export.write(shot, lib, RENDER)
                shots.append(shot.directory)
            names = ["scene/shot.usda", "scene/hair.usdc", "manifest.json"]
            _, mismatch, errors = filecmp.cmpfiles(shots[0], shots[1], names, shallow=False)
            self.assertEqual((mismatch, errors), ([], []))
            record = json.loads((shots[0] / "manifest.json").read_text())
            self.assertIsNotNone(record["scene"]["hero"]["hairstyle"])
            self.assertIn({"name": "hero_hair", "first_id": 4, "count": 1}, record["instance_ids"])
            stage = Usd.Stage.Open(str(shots[0] / "scene" / "shot.usda"))
            self.assertEqual(stage.GetPrimAtPath("/World/hero/hair/scalp").GetTypeName(), "BasisCurves")


if __name__ == "__main__":
    unittest.main()
