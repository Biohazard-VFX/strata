import unittest

import numpy as np
from pxr import Usd, UsdGeom

from strata_datagen.generators import camera, lighting
from strata_datagen.generators.terrain import build, sample_params
from strata_datagen.rng import stream

FRAMES = tuple(range(1, 25))


class CameraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ground = build(sample_params(stream(3, "t")), stream(3, "n"))
        self.target = np.array([0.0, 0.4, 0.0])

    def test_lens_values_are_tenths_of_a_scene_unit(self) -> None:
        track = camera.sample(
            stream(1, "c"),
            self.ground,
            self.target,
            1.6,
            frames=FRAMES,
            choices=camera.CameraChoices(move="locked"),
        )
        stage = Usd.Stage.CreateInMemory()
        cam = camera.author(stage, "/cam", track)
        self.assertAlmostEqual(
            cam.GetFocalLengthAttr().Get(Usd.TimeCode(1)), track.frame_lenses()[0, 0] / 100.0, places=4
        )
        self.assertAlmostEqual(cam.GetHorizontalApertureAttr().Get(), camera.SENSOR_MM[0] / 100.0, places=5)

    def test_fixed_choices_are_kept(self) -> None:
        choices = camera.CameraChoices(move="orbit", depth_of_field=True, motion_blur=False)
        track = camera.sample(stream(2, "c"), self.ground, self.target, 1.6, frames=FRAMES, choices=choices)
        self.assertEqual(track.move, "orbit")
        self.assertGreater(track.lens.fstop, 0.0)
        self.assertEqual(track.shutter, (0.0, 0.0))
        self.assertEqual(len(track.camera_to_world), len(FRAMES))

    def test_moves_keep_their_speed_whatever_the_clip_length(self) -> None:
        def angle(frames: tuple[int, ...]) -> float:
            choices = camera.CameraChoices(move="orbit")
            track = camera.sample(
                stream(4, "c"), self.ground, self.target, 1.6, frames=frames, choices=choices
            )
            eyes = np.array([m[:3, 3] - self.target for m in track.camera_to_world])
            first, last = eyes[0, [0, 2]], eyes[-1, [0, 2]]
            cos = first @ last / np.linalg.norm(first) / np.linalg.norm(last)
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) / (len(frames) - 1))

        self.assertAlmostEqual(angle(tuple(range(1, 9))), angle(FRAMES), delta=0.02)

    def test_every_move_projects_the_subject_into_frame(self) -> None:
        for move in (*camera.MOVES, camera.PATH):
            track = camera.sample(
                stream(4, move),
                self.ground,
                self.target,
                1.6,
                frames=FRAMES,
                choices=camera.CameraChoices(move=move),
            )
            for world_to_camera, (fx, fy, cx, cy) in zip(
                track.world_to_camera(), track.intrinsics(1280, 720), strict=True
            ):
                rotation = world_to_camera[:3, :3]
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
                x, y, z, _ = world_to_camera @ np.append(self.target, 1.0)
                self.assertLess(z, 0.0, move)
                self.assertLess(abs(fx * x / -z), cx, move)
                self.assertLess(abs(fy * y / -z), cy, move)


class SunTest(unittest.TestCase):
    def test_sun_light_axis_points_at_the_sun(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        hdri = lighting.Hdri("h", "h.exr", (0.48, 0.6, -0.64), 5.0, (1.0, 1.0, 1.0), 1.0)
        light = lighting.Lighting(hdri, rotation_deg=70.0, exposure_scale=1.0)
        lighting.author(stage, "/lights", light, 1)
        xform = UsdGeom.Xformable(stage.GetPrimAtPath("/lights/sun")).ComputeLocalToWorldTransform(0)
        axis = np.asarray(xform, dtype=np.float64)[2, :3]
        np.testing.assert_allclose(axis / np.linalg.norm(axis), light.sun_direction_world(), atol=1e-6)
        dome = stage.GetPrimAtPath("/lights/dome")
        self.assertTrue(dome.GetAttribute("inputs:karma:light:renderlightgeo").Get())

    def test_backdrop_is_camera_only_sky_mapped_like_the_dome(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        hdri = lighting.Hdri("h", "h.exr", (0.48, 0.6, -0.64), 5.0, (1.0, 1.0, 1.0), 1.0)
        lighting.author(stage, "/lights", lighting.Lighting(hdri, 0.0, 1.0), 1)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/lights/backdrop"))
        points = np.array(mesh.GetPointsAttr().Get())
        st = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get())
        centre = np.argmin(np.abs(st - [0.5, 0.5]).sum(axis=1))
        np.testing.assert_allclose(points[centre] / lighting.BACKDROP_M, [0.0, 0.0, 1.0], atol=1e-6)
        top = np.argmax(st[:, 1])
        np.testing.assert_allclose(points[top] / lighting.BACKDROP_M, [0.0, 1.0, 0.0], atol=1e-6)
        primvars = UsdGeom.PrimvarsAPI(mesh)
        self.assertEqual(primvars.GetPrimvar("karma:object:rendervisibility").Get(), "+primary")
        self.assertEqual(primvars.GetPrimvar("instance_id").Get(), lighting.SKY_ID)
        self.assertLess(lighting.BACKDROP_M, camera.CLIPPING_M[1])

    def test_lighting_drifts_smoothly_over_the_shot(self) -> None:
        hdri = lighting.Hdri("h", "h.exr", (0.48, 0.6, -0.64), 5.0, (1.0, 1.0, 1.0), 1.0)
        light = lighting.sample((hdri,), stream(5, "l"), 3.0, None, None, samples=722)
        stage = Usd.Stage.CreateInMemory()
        lighting.author(stage, "/lights", light, 1)
        dome = stage.GetPrimAtPath("/lights/dome").GetAttribute("inputs:intensity")
        values = np.array([dome.Get(Usd.TimeCode(f)) for f in range(722)])
        self.assertGreater(values.max() / values.min(), 1.2)
        self.assertLess(np.abs(np.diff(np.log2(values))).max(), 0.01)
        turn = stage.GetPrimAtPath("/lights").GetAttribute("xformOp:rotateY")
        angles = np.array([turn.Get(Usd.TimeCode(f)) for f in range(722)])
        self.assertGreater(np.ptp(angles), 5.0)
        self.assertLess(np.abs(np.diff(angles)).max(), 0.5)


if __name__ == "__main__":
    unittest.main()
