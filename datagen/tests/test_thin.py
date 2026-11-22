import itertools
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from pxr import Usd, UsdGeom, UsdShade
from scipy.spatial import KDTree

from strata_datagen.fields import Ground
from strata_datagen.generators import camera, thin
from strata_datagen.rng import stream
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import ShotSpec
from strata_datagen.usdutil import FloatArray
from tests.test_determinism import RENDER, hero_files, library

GROUND = Ground(np.zeros((201, 201)), 1.0, {})
FOCAL_MM = 35.0


def _view(eyes: FloatArray, subject: FloatArray | None = None) -> thin.View:
    subject = np.linspace([0.0, 0.0, 0.0], [0.2, 0.0, 0.1], len(eyes)) if subject is None else subject
    return thin.View(
        cameras=np.stack(
            [camera.look_at(e, s + np.array([0.0, 0.5, 0.0])) for e, s in zip(eyes, subject, strict=True)]
        ),
        tan_half=(0.5 * 36.0 / FOCAL_MM, 0.5 * 20.25 / FOCAL_MM),
        pixel_rad=36.0 / FOCAL_MM / 1280,
        subject=subject,
        subject_radius_m=0.6,
        subject_height_m=0.9,
    )


TIMES = tuple(range(49))
VIEW = _view(np.array([[0.0, 1.2, 7.0], [0.05, 1.2, 7.0], [0.1, 1.2, 7.0]]))


def _one(kind: str, band: str, seed: int) -> thin.Structure | None:
    found = thin.sample(stream(seed, "t"), GROUND, VIEW, (thin.Choice(kind, band),))
    return found[0] if found else None


def _structures() -> list[thin.Structure]:
    out: list[thin.Structure] = []
    for kind, spec in sorted(thin.KINDS.items()):
        for band in spec.bands:
            placed = [s for s in (_one(kind, band, seed) for seed in range(4)) if s is not None]
            if not placed:
                raise AssertionError(f"{kind} never placed in {band}")
            out.append(placed[0])
    return out


class CatenaryTest(unittest.TestCase):
    def test_level_span_hangs_by_its_sag_and_meets_both_ends(self) -> None:
        start, end = np.array([0.0, 10.0, 0.0]), np.array([30.0, 10.0, 40.0])
        cable = thin.catenary(start, end, 1.5, 201)
        np.testing.assert_allclose(cable[[0, -1]], [start, end], atol=1e-9)
        self.assertAlmostEqual(float(cable[:, 1].min()), 8.5, places=4)
        self.assertEqual(int(np.argmin(cable[:, 1])), 100)
        slope = np.gradient(cable[:, 1], np.linspace(0.0, 50.0, 201))
        curvature = np.gradient(slope, np.linspace(0.0, 50.0, 201))[5:-5]
        ratio = curvature / np.sqrt(1.0 + slope[5:-5] ** 2)
        self.assertLess(float(np.ptp(ratio) / ratio.mean()), 1e-2)

    def test_uneven_span_ends_on_both_anchors_with_its_low_point_near_the_lower_one(self) -> None:
        start, end = np.array([0.0, 4.0, 0.0]), np.array([20.0, 6.0, 0.0])
        cable = thin.catenary(start, end, 0.8, 101)
        np.testing.assert_allclose(cable[[0, -1]], [start, end], atol=1e-9)
        self.assertLess(int(np.argmin(cable[:, 1])), 50)
        self.assertLess(float(cable[:, 1].min()), 4.0)


class ShapeTest(unittest.TestCase):
    def test_chain_link_neighbours_hook_without_passing_through_each_other(self) -> None:
        fence = _one("chain_link", "between", 3)
        assert fence is not None
        wires = fence.strands[:-1]
        for left, right in itertools.pairwise(wires[:21]):
            gap = KDTree(thin.densify(left, 0.2 * fence.width_m)).query(
                thin.densify(right, 0.2 * fence.width_m)
            )
            self.assertGreaterEqual(float(gap[0].min()), 1.0 * fence.width_m)
            self.assertLess(float(gap[0].min()), 1.5 * fence.width_m)

    def test_laid_rope_strands_wind_around_the_centre_line(self) -> None:
        centre = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        strands = thin.twisted(centre, 0.02)
        self.assertEqual(len(strands), 3)
        for strand in strands:
            offset = np.linalg.norm(strand[:, 1:] - np.array([1.0, 0.0]), axis=1)
            np.testing.assert_allclose(offset, 0.006, atol=1e-9)
            turns = np.unwrap(np.arctan2(strand[:, 2], strand[:, 1] - 1.0))
            self.assertAlmostEqual(abs(float(turns[-1] - turns[0])) / (2 * np.pi), 2.0 / 0.07, delta=0.2)


class PlacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.structures = _structures()

    def test_every_kind_sits_in_its_band_clear_of_camera_and_subject_and_in_frame(self) -> None:
        eye = VIEW.eyes[1]
        subject = float(np.hypot(*(VIEW.subject[1] - eye)[[0, 2]]))
        for s in self.structures:
            with self.subTest(kind=s.kind, band=s.band):
                if s.band == "background":
                    self.assertGreater(s.distance_m, subject + VIEW.subject_radius_m)
                else:
                    self.assertLess(s.distance_m, subject - VIEW.subject_radius_m)
                points = s.points()
                self.assertGreaterEqual(float(KDTree(points).query(VIEW.eyes)[0].min()), 0.6 * s.distance_m)
                low = points[points[:, 1] < VIEW.subject_height_m][:, [0, 2]]
                if len(low):
                    near = KDTree(low).query(VIEW.subject[:, [0, 2]])[0].min()
                    self.assertGreater(float(near), VIEW.subject_radius_m)
                local = (np.linalg.inv(VIEW.cameras[s.anchor]) @ np.c_[points, np.ones(len(points))].T).T
                depth = -local[:, 2]
                inside = (np.abs(local[:, 0]) < depth * VIEW.tan_half[0]) & (
                    np.abs(local[:, 1]) < depth * VIEW.tan_half[1]
                )
                self.assertGreaterEqual(int((inside & (depth > 0)).sum()), 20)

    def test_widths_stay_physical_and_thin_on_screen(self) -> None:
        for s in self.structures:
            with self.subTest(kind=s.kind, band=s.band):
                low, high = thin.KINDS[s.kind].width_m
                self.assertTrue(low <= s.width_m <= high)
                self.assertAlmostEqual(s.width_px, s.width_m / (s.distance_m * VIEW.pixel_rad), places=6)
                self.assertTrue(s.width_px <= thin.PIXEL_WIDTH_RANGE[1] + 1e-9 or s.width_m == low)
        self.assertLess(min(s.width_px for s in self.structures), 1.0)

    def test_rigid_members_stay_within_a_few_dozen_pixels(self) -> None:
        for s in self.structures:
            footprint = s.distance_m * VIEW.pixel_rad
            for member in s.members:
                diameter_px = 2.0 * member.radius_m / footprint
                self.assertTrue(
                    diameter_px <= thin.MAX_MEMBER_PX + 1e-9 or member.radius_m <= thin.MIN_MEMBER_RADIUS_M
                )

    def test_fixing_one_structure_never_shifts_another(self) -> None:
        a = thin.sample(stream(5, "t"), GROUND, VIEW, (thin.Choice("net", "between"), thin.Choice()))
        b = thin.sample(stream(5, "t"), GROUND, VIEW, (thin.Choice("railing", "between"), thin.Choice()))
        self.assertEqual(a[-1].record(), b[-1].record())
        np.testing.assert_array_equal(a[-1].points(), b[-1].points())

    def test_an_impossible_band_raises(self) -> None:
        with self.assertRaises(ValueError):
            thin.sample(stream(1, "t"), GROUND, VIEW, (thin.Choice("antenna", "foreground"),))

    def test_authored_parts_are_valid_curves_and_meshes_with_their_own_ids(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        next_id = 100
        for index, s in enumerate(self.structures):
            parts = thin.author(
                stage, f"/thin/s{index}", s, next_id, name=f"s{index}", first_frame=1, samples=len(TIMES)
            )
            self.assertEqual([p.instance_id for p in parts], list(range(next_id, next_id + len(parts))))
            next_id += len(parts)
            total = 0
            for part in parts:
                for path in part.prims:
                    prim = stage.GetPrimAtPath(path)
                    self.assertEqual(
                        UsdGeom.PrimvarsAPI(prim).GetPrimvar("instance_id").Get(), part.instance_id
                    )
                    if part.name == "members":
                        mesh = UsdGeom.Mesh(prim)
                        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                        self.assertEqual(
                            int(np.asarray(mesh.GetFaceVertexCountsAttr().Get()).sum()), len(indices)
                        )
                        self.assertLess(int(indices.max()), len(mesh.GetPointsAttr().Get()))
                        continue
                    curves = UsdGeom.BasisCurves(prim)
                    counts = np.asarray(curves.GetCurveVertexCountsAttr().Get())
                    total += len(counts)
                    self.assertEqual(int(counts.sum()), len(curves.GetPointsAttr().Get()))
                    self.assertEqual(len(curves.GetWidthsAttr().Get()), int(counts.sum()))
                    transform = prim.GetAttribute("xformOp:transform")
                    samples = transform.GetNumTimeSamples() if transform else 0
                    self.assertEqual(samples, 0 if path.endswith("/still") else len(TIMES))
            self.assertEqual(total, len(s.strands))


class MotionTest(unittest.TestCase):
    def test_hanging_spans_move_smoothly_and_keep_their_anchors(self) -> None:
        swung = [
            w
            for kind in ("wires", "rigging", "net")
            for s in [_one(kind, "background", 2)]
            if s
            for w in s.swings
        ]
        self.assertTrue(swung)
        seconds = np.arange(0, 30 * 24) / 24.0
        for swing in swung:
            poses = swing.matrices(seconds)
            for anchor in (swing.start, swing.end):
                moved = poses[:, :3, :3] @ anchor + poses[:, :3, 3]
                np.testing.assert_allclose(moved, np.broadcast_to(anchor, moved.shape), atol=1e-9)
            step = np.abs(np.diff(poses[:, :3, :3], axis=0)).max()
            rate = swing.swing_rad * (swing.wind.max_slope + swing.gust.max_slope)
            bound = (rate + swing.sag_change * swing.tension.max_slope) / 24.0
            self.assertLess(float(step), bound + 1e-9)
            self.assertGreater(float(np.ptp(poses[:, 1, 1])), 0.0)

    def test_long_shots_anchor_structures_along_the_whole_path(self) -> None:
        eyes = np.linspace([-20.0, 1.2, 7.0], [20.0, 1.2, 7.0], 30 * 24 + 2)
        view = _view(eyes, eyes * np.array([1.0, 0.0, 0.0]))
        anchors = sorted(
            s.anchor for seed in range(6) for s in thin.sample(stream(seed, "long"), GROUND, view, None)
        )
        windows = {a // (24 * 8) for a in anchors}
        self.assertGreater(len(anchors), 6)
        self.assertGreaterEqual(len(windows), 3)


class ComposeTest(unittest.TestCase):
    def test_shots_author_thin_structures_with_instance_ids_in_the_manifest(self) -> None:
        lib = library()
        with TemporaryDirectory() as tmp:
            hero_files(Path(tmp), lib)
            spec = ShotSpec(
                seed=6, thin=(thin.Choice("chain_link", "between"), thin.Choice("wires", "background"))
            )
            shot = compose.plan(spec, lib, Path(tmp))
            compose.write(shot, lib, Path(tmp), RENDER)
            manifest = export.manifest(shot, lib, RENDER)
            stage = Usd.Stage.Open(str(shot.scene_dir / "shot.usda"))
            ranges = cast("list[dict[str, object]]", manifest["instance_ids"])
            ids = {r["name"]: r["first_id"] for r in ranges}
            records = cast("list[dict[str, object]]", cast("dict[str, object]", manifest["scene"])["thin"])
            for index, s in enumerate(shot.thin):
                group = stage.GetPrimAtPath(f"/World/thin/thin_{index}_{s.kind}/strands")
                prim = cast("list[Usd.Prim]", group.GetChildren())[0]  # pyright: ignore[reportUnknownMemberType]
                self.assertTrue(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])
                value = UsdGeom.PrimvarsAPI(prim).GetPrimvar("instance_id").Get()
                self.assertEqual(value, ids[f"thin_{index}_{s.kind}_strands"])
            self.assertEqual([r["kind"] for r in records], [s.kind for s in shot.thin])
            self.assertTrue(shot.thin)

    def test_a_seed_gives_byte_identical_thin_layers(self) -> None:
        lib = library()
        spec = ShotSpec(seed=11, thin=(thin.Choice("net"), thin.Choice("rigging"), thin.Choice()))
        with TemporaryDirectory() as tmp:
            layers: list[list[bytes]] = []
            for name in ("a", "b"):
                hero_files(Path(tmp) / name, lib)
                shot = compose.plan(spec, lib, Path(tmp) / name)
                compose.write(shot, lib, Path(tmp) / name, RENDER)
                layers.append([p.read_bytes() for p in sorted(shot.scene_dir.glob("*.usd*"))])
            self.assertTrue(
                b"@./thin_0_net.usdc@" in layers[0][-1] or any(b"thin_0_net" in x for x in layers[0])
            )
            self.assertGreater(len(layers[0]), 2)
            self.assertTrue(layers[0] == layers[1])


if __name__ == "__main__":
    unittest.main()
