import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen import regions
from strata_datagen.generators import feathers, groom
from strata_datagen.library import coat, creatures, heroes
from strata_datagen.rng import stream


def grid(n: int, size: float) -> groom.Skin:
    """A flat n x n quad grid in the xz plane facing +y, triangulated, with planar uvs."""
    u = np.linspace(0.0, size, n + 1)
    x, z = np.meshgrid(u, u, indexing="ij")
    points = np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=-1)
    tris: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = (
                i * (n + 1) + j,
                (i + 1) * (n + 1) + j,
                (i + 1) * (n + 1) + j + 1,
                i * (n + 1) + j + 1,
            )
            tris += [(a, c, b), (a, d, c)]
    triangles = np.asarray(tris, dtype=np.int64)
    st = (points[:, [0, 2]] / size)[triangles].reshape(-1, 2)
    return groom.Skin(points, triangles, st)


def params(clump_count: int = 0, clump_strength: float = 0.0) -> groom.GroomParams:
    return groom.GroomParams(400, 6, 0.0002, 0.00005, 0.05, 0.0, 0.0, 4, clump_count, clump_strength)


class GroomClumpTest(unittest.TestCase):
    def test_full_clumping_gathers_tips_on_leaders(self) -> None:
        skin = grid(8, 1.0)
        loose = groom.build(skin, None, params(), stream(1, "g"))
        tight = groom.build(skin, None, params(clump_count=10, clump_strength=1.0), stream(1, "g"))
        np.testing.assert_allclose(tight.curves[:, 0], loose.curves[:, 0])
        self.assertEqual(len(np.unique(tight.curves[:, -1].round(9), axis=0)), 10)

    def test_roots_carry_their_triangle_binding(self) -> None:
        skin = grid(4, 1.0)
        hair = groom.build(skin, None, params(), stream(2, "g"))
        anchor = np.einsum("nk,nkd->nd", hair.weights, skin.points[skin.triangles[hair.face]])
        np.testing.assert_allclose(anchor, hair.curves[:, 0], atol=1e-12)


class FeatherTest(unittest.TestCase):
    def test_feather_has_shaft_and_barbs_starting_on_it(self) -> None:
        p = feathers.FeatherParams(3, 0.04, 0.0, 0.01, 5, 45.0, 0.2, 0.3, 0.0005, 0.0001, 8)
        place = feathers.Placement(
            np.zeros((3, 3)),
            np.tile([0.0, 1.0, 0.0], (3, 1)),
            np.tile([0.0, 0.0, 1.0], (3, 1)),
            np.zeros((3, 2)),
        )
        built = feathers.build(place, p, stream(3, "f"))
        self.assertEqual(built.curves.shape, (3 * 11, 8, 3))
        shaft = built.curves.reshape(3, 11, 8, 3)[:, 0]
        barb_roots = built.curves.reshape(3, 11, 8, 3)[:, 1:, 0]
        distance = np.linalg.norm(barb_roots[:, :, None] - shaft[:, None], axis=-1).min(axis=-1)
        self.assertLess(float(distance.max()), 1e-9)
        sides = np.sign(built.curves.reshape(3, 11, 8, 3)[0, 1:, -1, 0])
        self.assertEqual(set(sides.tolist()), {-1.0, 1.0})


class CombTest(unittest.TestCase):
    def test_comb_points_away_from_the_front_and_survives_split_seams(self) -> None:
        skin = grid(10, 1.0)
        split = np.concatenate([skin.points, skin.points])
        triangles = skin.triangles.copy()
        triangles[len(triangles) // 2 :] += len(skin.points)
        seamed = groom.Skin(split, triangles, skin.corner_st)
        front = int(np.argmin(split[:, 0] + 10 * split[:, 2]))
        f = coat.field(seamed, {"body": np.ones(len(split))}, front)
        self.assertTrue(np.isfinite(f.distance[np.unique(triangles)]).all())
        used = np.unique(triangles)
        self.assertGreater(float(np.median(f.combs[used, 0] + f.combs[used, 2])), 0.9)
        np.testing.assert_allclose(f.normals[used, 1], 1.0)

    def test_deform_carries_hair_rigidly(self) -> None:
        skin = grid(4, 1.0)
        hair = groom.build(skin, None, params(), stream(4, "g"))
        angle = np.radians(70.0)
        turn = np.array(
            [[np.cos(angle), 0.0, -np.sin(angle)], [0.0, 1.0, 0.0], [np.sin(angle), 0.0, np.cos(angle)]]
        )
        moved = coat.deform(hair, skin.triangles, skin.points, skin.points @ turn + [1.0, 2.0, 3.0])
        np.testing.assert_allclose(moved, hair.curves @ turn + [1.0, 2.0, 3.0], atol=1e-9)


def box_raw(frames: int) -> creatures.Raw:
    """A flat body heading -X with a head region at that end, moving forward and rising each frame."""
    skin = grid(24, 1.0)
    top = skin.points * [0.3, 0.0, 1.0] + [0.0, 0.2, 0.0]
    points = top[:, [2, 1, 0]] * [-1.0, 1.0, 1.0]
    clip = np.stack([points + np.array([-0.05, 0.01, 0.0]) * f for f in range(frames)])
    names = ("head", "body")
    head = (top[skin.triangles].mean(axis=1)[:, 2] > 0.8).astype(np.int64)
    regions = np.where(head == 1, 0, 1)
    return creatures.Raw(clip, skin.triangles, skin.corner_st, regions, names, tuple(range(frames)), {})


class CreatureTest(unittest.TestCase):
    def test_orient_heads_plus_z_scales_and_drift_is_removed(self) -> None:
        raw = box_raw(4)
        oriented = creatures.orient(raw, None, 2.0)
        rest = oriented[0]
        self.assertAlmostEqual(float(rest[:, 2].max() - rest[:, 2].min()), 2.0)
        head = np.unique(raw.triangles[raw.regions == 0])
        self.assertGreater(float(rest[head, 2].mean()), 0.5)
        self.assertAlmostEqual(float(rest[:, 1].min()), 0.0)
        still = creatures.in_place(oriented)
        np.testing.assert_allclose(
            still[:, :, [0, 2]].mean(axis=1),
            still[0, :, [0, 2]].mean(axis=1)[None].repeat(4, axis=0),
            atol=1e-9,
        )
        self.assertGreater(float(still[-1, :, 1].mean() - still[0, :, 1].mean()), 0.0)

    def test_bare_creature_prepares_without_groom(self) -> None:
        raw = box_raw(1)
        rest = creatures.orient(raw, None, 1.0)[0]
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "bare"
            creatures.write_pose(folder / "poses" / "rest.usdc", raw, rest[None])
            skin = groom.Skin(rest, raw.triangles, raw.corner_st)
            front = creatures.front_vertex(raw, rest)
            field = coat.field(skin, creatures.region_weights(raw), front)
            hero = {
                "key": "bare",
                "poses": {"rest": "poses/rest.usdc"},
                "skin_prim": creatures.SKIN_PRIM,
                "albedo": "a.png",
                "coat": creatures.sample_coat(["bare"], field, rest, front, "bare"),
                "source": "test",
                "license": "test",
            }
            (folder / "hero.json").write_text(json.dumps(hero))
            prepared = coat.prepare_pose(heroes.load(folder), "rest", Path(tmp) / "rest.usdc")
            stage = Usd.Stage.Open(str(prepared.file))
            self.assertEqual(prepared.hairs, 0)
            self.assertTrue(stage.GetPrimAtPath("/hero/skin"))
            self.assertFalse(stage.GetPrimAtPath("/hero/groom"))

    def test_pose_skins_carry_soft_region_weights_that_recover_their_faces(self) -> None:
        raw = box_raw(1)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rest.usdc"
            creatures.write_pose(path, raw, creatures.orient(raw, None, 1.0)[:1])
            weights = coat.read_regions(Usd.Stage.Open(str(path)), creatures.SKIN_PRIM)
        self.assertEqual(sorted(weights), ["body", "head"])
        np.testing.assert_allclose(weights["head"] + weights["body"], 1.0)
        border = (weights["head"] > 0.0) & (weights["head"] < 1.0)
        self.assertTrue(border.any())
        counts, corners = np.full(len(raw.triangles), 3), raw.triangles.ravel()
        recovered = regions.face_mask(weights["head"], counts, corners)
        agree = float((recovered == (raw.regions == 0)).mean())
        self.assertGreater(agree, 0.97)

    def test_capped_coats_thin_out_and_thicken(self) -> None:
        raw = box_raw(1)
        rest = creatures.orient(raw, None, 1.0)[0]
        skin = groom.Skin(rest, raw.triangles, raw.corner_st)
        front = creatures.front_vertex(raw, rest)
        field = coat.field(skin, creatures.region_weights(raw), front)
        full = creatures.sample_coat(["short_coat"], field, rest, front, "box")
        with mock.patch.object(creatures, "MAX_HAIRS", 1000):
            capped = creatures.sample_coat(["short_coat"], field, rest, front, "box")
        for loose, tight in zip(
            cast("list[dict[str, dict[str, float]]]", full["fur"]),
            cast("list[dict[str, dict[str, float]]]", capped["fur"]),
            strict=True,
        ):
            self.assertLess(tight["groom"]["count"], loose["groom"]["count"])
            ratio = tight["groom"]["width_root_m"] / loose["groom"]["width_root_m"]
            self.assertGreater(ratio, 1.0)
            self.assertLessEqual(ratio, creatures.MAX_THICKEN + 1e-6)

    def test_animated_coat_pose_round_trip_is_deterministic(self) -> None:
        raw = box_raw(3)
        oriented = creatures.orient(raw, None, 1.0)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "box"
            creatures.write_pose(folder / "poses" / "walk.usdc", raw, creatures.in_place(oriented))
            skin = groom.Skin(oriented[0], raw.triangles, raw.corner_st)
            front = creatures.front_vertex(raw, oriented[0])
            field = coat.field(skin, creatures.region_weights(raw), front)
            layers = creatures.sample_coat(["short_coat", "whiskers"], field, oriented[0], front, "box")
            hero = {
                "key": "box",
                "poses": {"walk": "poses/walk.usdc"},
                "skin_prim": creatures.SKIN_PRIM,
                "albedo": "a.png",
                "coat": layers,
                "source": "test",
                "license": "test",
            }
            (folder / "hero.json").write_text(json.dumps(hero))
            manifest = heroes.load(folder)
            first = coat.prepare_pose(manifest, "walk", Path(tmp) / "a.usdc")
            second = coat.prepare_pose(manifest, "walk", Path(tmp) / "b.usdc")
            self.assertEqual((Path(tmp) / "a.usdc").read_bytes(), (Path(tmp) / "b.usdc").read_bytes())
            stage = Usd.Stage.Open(str(first.file))
            hair = UsdGeom.BasisCurves(stage.GetPrimAtPath("/hero/groom")).GetPointsAttr()
            self.assertEqual(list(hair.GetTimeSamples()), [0.0, 1.0, 2.0])
            self.assertEqual(first.hairs, second.hairs)
            lifted = np.asarray(hair.Get(2.0)) - np.asarray(hair.Get(0.0))
            self.assertGreater(float(lifted[:, 1].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
