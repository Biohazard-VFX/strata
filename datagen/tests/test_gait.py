import unittest

import numpy as np

from strata_datagen.fields import Ground
from strata_datagen.generators import hero
from strata_datagen.rng import stream
from strata_datagen.scene import creature
from strata_datagen.timeline import FPS
from strata_datagen.usdutil import FloatArray

PERIOD = 24
STRIDE_MPS = 1.2


def walker() -> FloatArray:
    """Two feet (blocks of points) and a body; each foot is planted half the cycle, sliding back.

    The body bobs, and in the frames where both feet swing the walker is airborne.
    """
    frames: list[FloatArray] = []
    for f in range(PERIOD + 1):
        points: list[FloatArray] = []
        for offset in (0, PERIOD // 2):
            u = ((f + offset) % PERIOD) / PERIOD
            planted = u < 0.4
            back = 0.3 - STRIDE_MPS * 0.4 * PERIOD / FPS
            z = 0.3 - STRIDE_MPS * u * PERIOD / FPS if planted else back + (0.3 - back) * (u - 0.4) / 0.6
            y = 0.0 if planted else 0.15 * np.sin(np.pi * (u - 0.4) / 0.6)
            points.append(np.array([[0.1, y, z], [0.12, y, z + 0.02], [0.1, y, z + 0.04]]))
        points.append(np.array([[0.0, 0.5 + 0.02 * np.cos(2 * np.pi * f / PERIOD), 0.0]]))
        frames.append(np.concatenate(points))
    return np.stack(frames)


def flapper() -> FloatArray:
    frames = [
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0 - 0.8 * np.sin(np.pi * f / PERIOD), 0.0]])
        for f in range(PERIOD + 1)
    ]
    return np.stack(frames)


def slope() -> Ground:
    x = np.linspace(-200.0, 200.0, 401)
    return Ground(np.tile(0.1 * x, (len(x), 1)), 1.0, {})


class GaitTest(unittest.TestCase):
    def test_stance_feet_set_the_speed_and_touch_the_ground(self) -> None:
        walk = creature.gait(walker())
        self.assertTrue(walk.grounded)
        self.assertFalse(walk.flying)
        self.assertAlmostEqual(float(np.median(walk.speed[walk.stance])), STRIDE_MPS, places=6)
        low = walker()[:-1, :, 1].min(axis=1)
        np.testing.assert_allclose((walk.lift + low)[walk.stance], 0.0, atol=1e-12)
        self.assertTrue(np.all(walk.lift + low >= -1e-12))

    def test_wing_beats_fly_and_idle_clips_stand(self) -> None:
        self.assertTrue(creature.gait(flapper()).flying)
        still = np.repeat(walker()[:1], PERIOD + 1, axis=0)
        idle = creature.gait(still)
        self.assertFalse(idle.grounded or idle.flying)

    def test_planted_feet_do_not_slide_and_stay_on_sloped_terrain(self) -> None:
        cycle, terrain = walker(), slope()
        walk = creature.gait(cycle)
        samples = 3 * PERIOD + 2
        drawn = hero.sample_motion(stream(3, "m"), terrain, samples, 0.0, 0.4)
        motion = creature.ground(drawn, walk, terrain, stream(3, "g"))
        angle = np.radians(motion.yaw_deg)
        c, s = np.cos(angle), np.sin(angle)
        world: list[FloatArray] = []
        for i in range(samples):
            p = cycle[i % PERIOD]
            world.append(
                np.stack([c[i] * p[:, 0] + s[i] * p[:, 2], p[:, 1], -s[i] * p[:, 0] + c[i] * p[:, 2]], -1)
                + motion.positions[i]
            )
        points: FloatArray = np.stack(world)
        clearance = points[..., 1] - terrain.height(points[..., 0], points[..., 2])
        self.assertGreater(float(clearance.min()), -1e-6)
        planted = np.flatnonzero(walk.stance[np.arange(samples - 1) % PERIOD])
        foot = 0 if walk.stance[0] else 3
        slide = np.linalg.norm(np.diff(points[:, foot, [0, 2]], axis=0), axis=1)[planted] * FPS
        self.assertLess(float(np.median(slide)), 0.1 * STRIDE_MPS)


class HeroPathTest(unittest.TestCase):
    def test_turning_paths_keep_their_speed_over_long_shots(self) -> None:
        for seed in range(6):
            motion = hero.sample_motion(stream(seed, "m"), slope(), 722, 0.0, 1.0)
            steps = np.linalg.norm(np.diff(motion.positions[:, [0, 2]], axis=0), axis=1) * FPS
            np.testing.assert_allclose(steps, 1.0, rtol=1e-9)
            np.testing.assert_allclose(motion.positions[361, [0, 2]], 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
