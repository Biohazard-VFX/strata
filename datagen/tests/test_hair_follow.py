import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.generators import hairstyle, human, materials
from strata_datagen.library import human_usd
from strata_datagen.library.catalogue import HeroHair
from strata_datagen.rng import stream
from strata_datagen.scene import hair
from strata_datagen.usdutil import FloatArray, new_stage
from tests.test_hairstyle import CENTRE, regions, sphere

FRAMES = 73


def posed(rest: FloatArray, times: FloatArray) -> FloatArray:
    """The test clip per time code: the head walks, turns, grows and stretches taller (not a similarity)."""
    centre = CENTRE + np.stack([0.02 * times, 0.01 * np.sin(times / 5.0), 0.01 * times], axis=-1)
    scale, stretch, yaw = 1.0 + 0.15 * times / FRAMES, 1.0 + 0.2 * times / FRAMES, np.radians(1.5 * times)
    c, s = np.cos(yaw), np.sin(yaw)
    out: list[FloatArray] = []
    for k in range(len(times)):
        turn = np.array([[c[k], 0.0, -s[k]], [0.0, 1.0, 0.0], [s[k], 0.0, c[k]]])
        out.append(centre[k] + scale[k] * ((rest - CENTRE) * [1.0, stretch[k], 1.0]) @ turn)
    return np.stack(out)


def clip(folder: Path) -> Path:
    points, triangles = sphere()
    times = np.array([f + o for f in range(FRAMES + 2) for o in (-0.25, 0.0, 0.25)])
    part = human_usd.MeshPart(
        "body", np.full(len(triangles), 3), triangles.ravel(), np.zeros((triangles.size, 2)), points, "skin"
    )
    path = folder / "clip.usdc"
    human_usd.write(
        path,
        parts=[part],
        samples={"body": posed(points, times)},
        times=times,
        looks={"skin": materials.Look(materials.Textures("albedo.png"))},
        regions={k: np.clip(v, 0.0, 1.0) for k, v in regions(points).items()},
    )
    return path


class FollowTest(unittest.TestCase):
    def test_similarity_recovers_a_scaled_rigid_motion(self) -> None:
        rng = np.random.default_rng(3)
        source = rng.normal(size=(50, 3))
        angle = np.radians(70.0)
        turn = np.array(
            [[np.cos(angle), np.sin(angle), 0.0], [-np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
        )
        target = 1.3 * source @ turn + [0.5, -2.0, 4.0]
        matrix = hair.similarity(source, target)
        np.testing.assert_allclose(hair.apply(matrix, source), target, atol=1e-9)
        np.testing.assert_allclose(hair.unapply(matrix, target), source, atol=1e-9)

    def test_hair_roots_stay_on_a_moving_growing_head_at_every_sample(self) -> None:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            asset = clip(folder)
            stage = new_stage(folder / "shot.usda")
            frames = tuple(range(1001, 1001 + 48))
            placement = human.ClipPlacement(
                10, 0.0, np.zeros((len(frames) + 2, 3)), np.zeros((len(frames) + 2, 3))
            )
            human.author(
                stage,
                "/World/hero",
                "./clip.usdc",
                placement,
                first_frame=frames[0],
                instance_id=2,
                name="hero_skin",
            )
            info = HeroHair(
                hairstyle.REGIONS,
                (0.0, 0.0, 1.0),
                facial_hair=False,
                skin_prim="/hero/body",
                rest_primvar="rest",
            )
            style = hairstyle.sample(
                stream(5, "hairstyle"), hairstyle.REGIONS, facial_hair=False, duration_s=2.0
            )
            hair.author(stage, folder, asset, info, style, frames=frames, placement=placement, seed=5)
            stage.GetRootLayer().Save()
            head, _ = hair.rest_head(asset, info)
            rest = (
                hairstyle.build(head, style, stream(5, "hairstyle", "build"), hairstyle.key_seconds(47 / 24))
                .sets[0]
                .curves[0, :, 0]
            )
            shot = Usd.Stage.Open(str(folder / "shot.usda"))
            curves = UsdGeom.BasisCurves(shot.GetPrimAtPath("/World/hero/hair/scalp"))
            counts = np.asarray(curves.GetCurveVertexCountsAttr().Get())
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            cache = UsdGeom.XformCache()
            worst, at_keys = 0.0, 0.0
            for code in np.arange(frames[0] - 1.25, frames[-1] + 1.5, 0.25):
                cache.SetTime(Usd.TimeCode(float(code)))
                world: FloatArray = np.asarray(
                    cache.GetLocalToWorldTransform(curves.GetPrim()), dtype=np.float64
                )
                roots: FloatArray = np.asarray(curves.GetPointsAttr().Get(float(code)), dtype=np.float64)[
                    starts
                ]
                roots = roots @ world[:3, :3] + world[3, :3]
                want: FloatArray = posed(rest, np.array([float(code) - frames[0] + placement.start]))[0]
                distance: FloatArray = np.linalg.norm(roots - want, axis=1)
                gap = float(distance.max())
                worst = max(worst, gap)
                at_keys = max(at_keys, gap) if code in (frames[0], frames[-1]) else at_keys
            self.assertLess(at_keys, 1e-4)
            self.assertLess(worst, 1e-3)


if __name__ == "__main__":
    unittest.main()
