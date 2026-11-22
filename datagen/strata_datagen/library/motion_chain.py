"""Long continuous motion: CMU clip windows chained with crossfades into one take.

Each window is turned and moved so it starts where the previous one ends (steering back towards
the origin once the walker is far from it), then blended over a crossfade: bone world rotations
are interpolated as quaternions and root positions linearly, with a smoothstep weight. Root
travel is rescaled to the first window's hip height so subjects of different sizes chain without jumps.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from strata_datagen.library import mocap
from strata_datagen.usdutil import FloatArray

CROSSFADE = 72
STEER_RADIUS_M = 4.0
MAX_STEER_DEG = 70.0


@dataclass(frozen=True)
class Window:
    trial: str
    start: int
    frames: int


def heading_deg(root_rotation: FloatArray) -> float:
    forward = root_rotation @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(forward[0], forward[2])))


def yaw(degrees: float) -> FloatArray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _continuous(quats: FloatArray) -> FloatArray:
    """Flip quaternion signs along time (axis 0) so consecutive samples lie in one hemisphere."""
    out = quats.copy()
    for i in range(1, len(out)):
        sign = np.where((out[i] * out[i - 1]).sum(axis=-1) < 0.0, -1.0, 1.0)
        out[i] = out[i] * sign[..., None]
    return out


def blend(a: FloatArray, b: FloatArray, w: FloatArray) -> FloatArray:
    """Crossfade rotation stacks (frames, bones, 3, 3) by normalised quaternion interpolation.

    Both tracks are made sign-continuous in time and `b` is matched to `a` at the first frame, so
    the result moves continuously even where the two poses differ by large angles.
    """
    shape = a.shape
    qa = _continuous(Rotation.from_matrix(a.reshape(-1, 3, 3)).as_quat().reshape(*shape[:2], 4))
    qb = _continuous(Rotation.from_matrix(b.reshape(-1, 3, 3)).as_quat().reshape(*shape[:2], 4))
    match = np.where((qa[0] * qb[0]).sum(axis=-1) < 0.0, -1.0, 1.0)
    qb = qb * match[None, :, None]
    q = qa * (1.0 - w)[:, None, None] + qb * w[:, None, None]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(shape)


def _hip_height(skeleton: mocap.Skeleton) -> float:
    return float(
        -sum(
            skeleton.bones[b].direction[1] * skeleton.bones[b].length_m
            for b in ("lhipjoint", "lfemur", "ltibia")
        )
    )


def chain(cmu: Path, windows: list[Window]) -> tuple[mocap.Skeleton, mocap.Motion]:
    loaded: dict[str, tuple[mocap.Skeleton, mocap.Motion]] = {}
    reference: mocap.Skeleton | None = None
    rotations = np.zeros((0, 0, 3, 3))
    root = np.zeros((0, 3))
    bones: tuple[str, ...] = ()
    for window in windows:
        if window.trial not in loaded:
            loaded[window.trial] = mocap.load(
                cmu / f"{window.trial.split('_')[0]}.asf", cmu / f"{window.trial}.amc"
            )
        skeleton, motion = loaded[window.trial]
        reference = reference or skeleton
        bones = motion.bones
        span = slice(window.start, window.start + window.frames)
        rot = motion.rotations[span]
        pos = motion.heads[span, 0] * (_hip_height(reference) / _hip_height(skeleton))
        if len(root) == 0:
            rotations, root = rot, pos - pos[0] * np.array([1.0, 0.0, 1.0])
            continue
        anchor = len(root) - CROSSFADE
        turn = heading_deg(rotations[anchor, 0]) - heading_deg(rot[0, 0])
        here = root[anchor]
        if np.hypot(here[0], here[2]) > STEER_RADIUS_M:
            home = float(np.degrees(np.arctan2(-here[0], -here[2])))
            current = heading_deg(rotations[anchor, 0])
            turn += float(np.clip((home - current + 180.0) % 360.0 - 180.0, -MAX_STEER_DEG, MAX_STEER_DEG))
        r = yaw(turn)
        rot = np.einsum("ij,fbjk->fbik", r, rot)
        pos = (pos - pos[0] * np.array([1.0, 0.0, 1.0])) @ r.T + here * np.array([1.0, 0.0, 1.0])
        w = np.linspace(0.0, 1.0, CROSSFADE)
        w = w * w * (3.0 - 2.0 * w)
        blended = blend(rotations[anchor:], rot[:CROSSFADE], w)
        blended_root = root[anchor:] * (1.0 - w[:, None]) + pos[:CROSSFADE] * w[:, None]
        rotations = np.concatenate([rotations[:anchor], blended, rot[CROSSFADE:]])
        root = np.concatenate([root[:anchor], blended_root, pos[CROSSFADE:]])
    if reference is None:
        raise ValueError("no motion windows")
    heads = np.zeros((*rotations.shape[:2], 3))
    heads[:, 0] = root
    return reference, mocap.Motion(bones, rotations, heads, heads)


def plan(
    rng: np.random.Generator, takes: dict[str, list[tuple[str, int]]], lengths: dict[str, int], seconds: float
) -> list[Window]:
    """Windows of 2.5-6 s from seeded categories until the chain covers `seconds` (crossfades included)."""
    categories = sorted(takes)
    total = 0
    out: list[Window] = []
    need = int(np.ceil(seconds * mocap.FPS))
    while total < need + CROSSFADE:
        category = categories[int(rng.integers(0, len(categories)))]
        trial, first = takes[category][int(rng.integers(0, len(takes[category])))]
        frames = min(int(rng.uniform(2.5, 6.0) * mocap.FPS), lengths[trial] - first - 1)
        if frames <= 2 * CROSSFADE:
            continue
        start = first + int(rng.integers(0, lengths[trial] - first - frames))
        out.append(Window(trial, start, frames))
        total += frames - (CROSSFADE if len(out) > 1 else 0)
    return out
