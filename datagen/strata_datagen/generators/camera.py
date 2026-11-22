"""Camera rigs: a continuous spline path for long shots, fixed moves for short ones, per-frame lens."""

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom

from strata_datagen import timeline
from strata_datagen.fields import Ground
from strata_datagen.usdutil import FloatArray, matrix

VERSION = "camera-0.2.0"
MOVES = ("locked", "pan_tilt", "handheld", "dolly", "orbit")
PATH = "path"
PATH_MIN_FRAMES = 97
SENSOR_MM = (36.0, 20.25)
CLIPPING_M = (0.03, 5000.0)
REFERENCE_SUBJECT_M = 1.6
MM_PER_LENS_UNIT = 100.0  # UsdGeomCamera lens values are tenths of a scene unit; Karma honours it.


@dataclass(frozen=True)
class Lens:
    focal_mm: float
    fstop: float
    focus_m: float


@dataclass(frozen=True)
class CameraTrack:
    """Camera_to_world matrices (column-vector convention; camera looks down -Z, +Y up).

    samples holds one pose per frame plus a handle frame before and after, for motion blur;
    lenses holds (focal_mm, fstop, focus_m) for the same samples. fstop 0 means no depth of field.
    """

    move: str
    frames: tuple[int, ...]
    samples: FloatArray
    lenses: FloatArray
    shutter: tuple[float, float]

    @property
    def camera_to_world(self) -> FloatArray:
        return self.samples[1:-1]

    @property
    def motion_blur(self) -> bool:
        return self.shutter[1] > self.shutter[0]

    @property
    def depth_of_field(self) -> bool:
        return bool((self.lenses[:, 1] > 0.0).any())

    @property
    def lens(self) -> Lens:
        """The lens at the middle frame, for summaries."""
        focal, fstop, focus = self.lenses[len(self.lenses) // 2]
        return Lens(float(focal), float(fstop), float(focus))

    def frame_lenses(self) -> FloatArray:
        return self.lenses[1:-1]

    def world_to_camera(self) -> FloatArray:
        return np.linalg.inv(self.camera_to_world)

    def intrinsics(self, width: int, height: int) -> FloatArray:
        """Per frame (fx, fy, cx, cy) in pixels."""
        focal = self.frame_lenses()[:, 0]
        return np.stack(
            [
                focal / SENSOR_MM[0] * width,
                focal / SENSOR_MM[1] * height,
                np.full_like(focal, 0.5 * width),
                np.full_like(focal, 0.5 * height),
            ],
            axis=-1,
        )


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


def _path(
    rng: np.random.Generator,
    ground: Ground,
    targets: FloatArray,
    subject_m: float,
    *,
    azimuth: float,
    dof: bool,
) -> tuple[FloatArray, FloatArray]:
    """A camera that never stops: azimuth, distance, height, framing and aperture drift on smooth curves."""
    t = timeline.sample_times(len(targets))
    duration = float(t[-1]) + 1.0
    swing = timeline.curve(rng, duration, 6.0, -60.0, 60.0)(t) + azimuth
    size = subject_m / REFERENCE_SUBJECT_M
    distance = size * np.exp(timeline.curve(rng, duration, 8.0, np.log(2.5), np.log(11.0))(t))
    height = size * timeline.curve(rng, duration, 7.0, 0.25, 2.2)(t)
    fill = timeline.curve(rng, duration, 9.0, 0.3, 0.9)(t)
    fstop = timeline.curve(rng, duration, 10.0, 1.4, 16.0)(t) if dof else np.zeros_like(t)
    poses = np.stack(
        [
            look_at(_eye(ground, target, float(a), float(d), float(h)), target)
            for target, a, d, h in zip(targets, swing, distance, height, strict=True)
        ]
    )
    focal = np.clip(SENSOR_MM[0] * distance * fill / subject_m, 18.0, 135.0)
    focus = np.linalg.norm(poses[:, :3, 3] - targets, axis=1)
    return poses, np.stack([focal, fstop, focus], axis=-1)


def sample(
    rng: np.random.Generator,
    ground: Ground,
    target: FloatArray,
    subject_m: float,
    *,
    frames: tuple[int, ...],
    choices: CameraChoices,
) -> CameraTrack:
    """Every value is drawn even when fixed, so fixing one choice never shifts the others.

    target is one point or one point per sample (frames plus two handles); long shots follow it on
    a spline path, short ones frame its middle sample with a fixed lens.
    """
    count = len(frames) + 2
    targets = np.broadcast_to(np.asarray(target, np.float64), (count, 3)).copy()
    drawn_move = str(rng.choice(MOVES))
    move = _pick(choices.move, PATH if len(frames) >= PATH_MIN_FRAMES else drawn_move)
    size = subject_m / REFERENCE_SUBJECT_M
    distance = _pick(choices.distance_m, size * float(np.exp(rng.uniform(np.log(2.5), np.log(11.0)))))
    height = _pick(choices.height_m, size * float(rng.uniform(0.25, 2.2)))
    azimuth = _pick(choices.azimuth_deg, float(rng.uniform(0.0, 360.0)))
    fill = _pick(choices.fill, float(rng.uniform(0.3, 0.9)))
    dof = _pick(choices.depth_of_field, bool(rng.random() < 0.4))
    blur = _pick(choices.motion_blur, bool(rng.random() < 0.4))
    fstop = float(rng.uniform(1.4, 5.6))
    shutter = (-0.25, 0.25) if blur else (0.0, 0.0)
    if move == PATH:
        poses, lenses = _path(rng, ground, targets, subject_m, azimuth=azimuth, dof=dof)
        return CameraTrack(move, frames, poses, lenses, shutter)
    middle_target = targets[count // 2]
    focal = float(np.clip(SENSOR_MM[0] * distance * fill / subject_m, 18.0, 135.0))
    aim = middle_target + rng.normal(0.0, 0.08 * subject_m, 3)
    t = np.linspace(0.0, 1.0, count)
    span = (len(frames) + 1) / 25.0
    poses = _poses(move, rng, ground, aim, t=t, span=span, azimuth=azimuth, distance=distance, height=height)
    offset: FloatArray = poses[len(poses) // 2][:3, 3] - middle_target
    middle = float(np.linalg.norm(offset))
    lenses = np.tile([focal, fstop if dof else 0.0, middle], (count, 1))
    return CameraTrack(move, frames, poses, lenses, shutter)


def author(stage: Usd.Stage, path: str, track: CameraTrack) -> UsdGeom.Camera:
    """Time-sampled transform at every frame plus one handle frame on each side for motion blur."""
    camera = UsdGeom.Camera.Define(stage, path)
    op = camera.AddTransformOp()
    first = track.frames[0] - 1
    for index, pose in enumerate(track.samples):
        op.Set(matrix(pose.T), Usd.TimeCode(first + index))
    focal, fstop, focus = (
        camera.CreateFocalLengthAttr(),
        camera.CreateFStopAttr(),
        camera.CreateFocusDistanceAttr(),
    )
    for index, lens in enumerate(track.lenses):
        time = Usd.TimeCode(first + index)
        focal.Set(float(lens[0]) / MM_PER_LENS_UNIT, time)
        fstop.Set(float(lens[1]), time)
        focus.Set(float(lens[2]), time)
    camera.CreateHorizontalApertureAttr(SENSOR_MM[0] / MM_PER_LENS_UNIT)
    camera.CreateVerticalApertureAttr(SENSOR_MM[1] / MM_PER_LENS_UNIT)
    nearest = float(track.lenses[:, 2].min())
    camera.CreateClippingRangeAttr(Gf.Vec2f(min(CLIPPING_M[0], 0.1 * nearest), CLIPPING_M[1]))
    camera.CreateShutterOpenAttr(track.shutter[0])
    camera.CreateShutterCloseAttr(track.shutter[1])
    return camera
