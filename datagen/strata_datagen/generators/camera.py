"""Camera rigs and moves: locked, nodal pan/tilt, handheld, dolly and orbit, with lens and shutter."""

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom

from strata_datagen.fields import Ground
from strata_datagen.usdutil import FloatArray, matrix

VERSION = "camera-0.1.0"
MOVES = ("locked", "pan_tilt", "handheld", "dolly", "orbit")
SENSOR_MM = (36.0, 20.25)
CLIPPING_M = (0.03, 5000.0)
MM_PER_LENS_UNIT = 100.0  # UsdGeomCamera lens values are tenths of a scene unit; Karma honours it.


@dataclass(frozen=True)
class Lens:
    focal_mm: float
    fstop: float
    focus_m: float


@dataclass(frozen=True)
class CameraTrack:
    """Camera_to_world matrices (column-vector convention; camera looks down -Z, +Y up).

    samples holds one pose per frame plus a handle frame before and after, for motion blur.
    """

    move: str
    frames: tuple[int, ...]
    samples: FloatArray
    lens: Lens
    shutter: tuple[float, float]

    @property
    def camera_to_world(self) -> FloatArray:
        return self.samples[1:-1]

    @property
    def motion_blur(self) -> bool:
        return self.shutter[1] > self.shutter[0]

    @property
    def depth_of_field(self) -> bool:
        return self.lens.fstop > 0.0

    def world_to_camera(self) -> FloatArray:
        return np.linalg.inv(self.camera_to_world)

    def intrinsics(self, width: int, height: int) -> tuple[float, float, float, float]:
        fx = self.lens.focal_mm / SENSOR_MM[0] * width
        fy = self.lens.focal_mm / SENSOR_MM[1] * height
        return fx, fy, 0.5 * width, 0.5 * height


def look_at(eye: FloatArray, target: FloatArray) -> FloatArray:
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = right, up, -forward, eye
    return out


def _rotate_y(vector: FloatArray, degrees: float) -> FloatArray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * vector[0] + s * vector[2], vector[1], -s * vector[0] + c * vector[2]])


def _smooth_noise(rng: np.random.Generator, t: FloatArray, dims: int) -> FloatArray:
    """Band-limited wobble in about [-1, 1]: three random sinusoids per dimension."""
    freq = rng.uniform(0.4, 3.5, (dims, 3))
    phase = rng.uniform(0.0, 2.0 * np.pi, (dims, 3))
    weight = rng.uniform(0.3, 1.0, (dims, 3))
    waves = np.sin(2.0 * np.pi * freq[None] * t[:, None, None] + phase[None]) * weight[None]
    return waves.sum(axis=2) / weight.sum(axis=1)[None]


def _eye(
    ground: Ground, target: FloatArray, azimuth_deg: float, distance: float, height: float
) -> FloatArray:
    flat = _rotate_y(np.array([0.0, 0.0, 1.0]), azimuth_deg) * distance
    x, z = target[0] + flat[0], target[2] + flat[2]
    y = float(ground.height(np.array([x]), np.array([z]))[0]) + height
    return np.array([x, y, z])


def _poses(
    move: str,
    rng: np.random.Generator,
    ground: Ground,
    target: FloatArray,
    *,
    t: FloatArray,
    span: float,
    azimuth: float,
    distance: float,
    height: float,
) -> FloatArray:
    if move == "orbit":
        sweep = rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 30.0) * span
        return np.stack(
            [look_at(_eye(ground, target, azimuth + sweep * s, distance, height), target) for s in t]
        )
    eye = _eye(ground, target, azimuth, distance, height)
    if move == "dolly":
        travel = _rotate_y(np.array([0.0, 0.0, 1.0]), rng.uniform(0.0, 360.0)) * rng.uniform(0.4, 2.0) * span
        return np.stack([look_at(eye + travel * (s - 0.5), target) for s in t])
    if move == "pan_tilt":
        pan, tilt = rng.uniform(-18.0, 18.0) * span, rng.uniform(-5.0, 5.0) * span
        aims = [target + np.array([0.0, np.tan(np.radians(tilt * (s - 0.5))) * distance, 0.0]) for s in t]
        return np.stack(
            [
                look_at(eye, eye + _rotate_y(aim - eye, pan * (s - 0.5)))
                for s, aim in zip(t, aims, strict=True)
            ]
        )
    base = look_at(eye, target)
    if move == "locked":
        return np.repeat(base[None], t.shape[0], axis=0)
    return _handheld(base, rng, t)


def _handheld(base: FloatArray, rng: np.random.Generator, t: FloatArray) -> FloatArray:
    seconds = t * (t.shape[0] - 1) / 24.0
    shake = _smooth_noise(rng, seconds, 6)
    position_m, angle_deg = rng.uniform(0.005, 0.03), rng.uniform(0.2, 1.2)
    poses: list[FloatArray] = []
    for row in shake:
        rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), float(row[3] * angle_deg))
        rotation *= Gf.Rotation(Gf.Vec3d(0, 1, 0), float(row[4] * angle_deg))
        rotation *= Gf.Rotation(Gf.Vec3d(0, 0, 1), float(row[5] * angle_deg * 0.5))
        local = np.array(Gf.Matrix4d().SetRotate(rotation)).T
        pose = base @ local
        pose[:3, 3] += row[:3] * position_m
        poses.append(pose)
    return np.stack(poses)


@dataclass(frozen=True)
class CameraChoices:
    """Fixed values for a shot; None means drawn from the seed."""

    move: str | None = None
    distance_m: float | None = None
    height_m: float | None = None
    azimuth_deg: float | None = None
    fill: float | None = None
    depth_of_field: bool | None = None
    motion_blur: bool | None = None


def _pick[T](fixed: T | None, drawn: T) -> T:
    return drawn if fixed is None else fixed


def sample(
    rng: np.random.Generator,
    ground: Ground,
    target: FloatArray,
    subject_m: float,
    *,
    frames: tuple[int, ...],
    choices: CameraChoices,
) -> CameraTrack:
    """Every value is drawn even when fixed, so fixing one choice never shifts the others."""
    move = _pick(choices.move, str(rng.choice(MOVES)))
    distance = _pick(choices.distance_m, float(np.exp(rng.uniform(np.log(2.5), np.log(11.0)))))
    height = _pick(choices.height_m, float(rng.uniform(0.25, 2.2)))
    azimuth = _pick(choices.azimuth_deg, float(rng.uniform(0.0, 360.0)))
    fill = _pick(choices.fill, float(rng.uniform(0.3, 0.9)))
    dof = _pick(choices.depth_of_field, bool(rng.random() < 0.4))
    blur = _pick(choices.motion_blur, bool(rng.random() < 0.4))
    fstop = float(rng.uniform(1.4, 5.6))
    focal = float(np.clip(SENSOR_MM[0] * distance * fill / subject_m, 18.0, 135.0))
    aim = target + rng.normal(0.0, 0.08 * subject_m, 3)
    t = np.linspace(0.0, 1.0, len(frames) + 2)
    span = (len(frames) + 1) / 25.0
    poses = _poses(move, rng, ground, aim, t=t, span=span, azimuth=azimuth, distance=distance, height=height)
    middle: FloatArray = poses[len(poses) // 2][:3, 3] - target
    lens = Lens(focal, fstop if dof else 0.0, float(np.linalg.norm(middle)))
    return CameraTrack(move, frames, poses, lens, (-0.25, 0.25) if blur else (0.0, 0.0))


def author(stage: Usd.Stage, path: str, track: CameraTrack) -> UsdGeom.Camera:
    """Time-sampled transform at every frame plus one handle frame on each side for motion blur."""
    camera = UsdGeom.Camera.Define(stage, path)
    op = camera.AddTransformOp()
    first = track.frames[0] - 1
    for index, pose in enumerate(track.samples):
        op.Set(matrix(pose.T), Usd.TimeCode(first + index))
    camera.CreateFocalLengthAttr(track.lens.focal_mm / MM_PER_LENS_UNIT)
    camera.CreateHorizontalApertureAttr(SENSOR_MM[0] / MM_PER_LENS_UNIT)
    camera.CreateVerticalApertureAttr(SENSOR_MM[1] / MM_PER_LENS_UNIT)
    camera.CreateFStopAttr(track.lens.fstop)
    camera.CreateFocusDistanceAttr(track.lens.focus_m)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*CLIPPING_M))
    camera.CreateShutterOpenAttr(track.shutter[0])
    camera.CreateShutterCloseAttr(track.shutter[1])
    return camera
