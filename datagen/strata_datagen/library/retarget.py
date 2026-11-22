"""Retarget CMU mocap onto MPFB's cmu_mb rig and skin meshes with linear blend skinning.

Each target bone takes the world rotation of its CMU bone, after a rest-pose swing that aligns
the target's rest direction with the CMU T-pose direction. Joint positions keep the target's own
proportions; root translation is scaled by the ratio of hip heights. Hand bones follow the wrist
because CMU finger channels are placeholders.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from strata_datagen.library.mocap import FPS, Motion, Skeleton
from strata_datagen.usdutil import FloatArray, IntArray

CMU_MB = {
    "Hips": "root",
    "LHipJoint": "lhipjoint",
    "LeftUpLeg": "lfemur",
    "LeftLeg": "ltibia",
    "LeftFoot": "lfoot",
    "LeftToeBase": "ltoes",
    "RHipJoint": "rhipjoint",
    "RightUpLeg": "rfemur",
    "RightLeg": "rtibia",
    "RightFoot": "rfoot",
    "RightToeBase": "rtoes",
    "LowerBack": "lowerback",
    "Spine": "upperback",
    "Spine1": "thorax",
    "Neck": "lowerneck",
    "Neck1": "upperneck",
    "Head": "head",
    "LeftShoulder": "lclavicle",
    "LeftArm": "lhumerus",
    "LeftForeArm": "lradius",
    "LeftHand": "lwrist",
    "LeftFingerBase": "lwrist",
    "LeftHandFinger1": "lwrist",
    "LThumb": "lwrist",
    "RightShoulder": "rclavicle",
    "RightArm": "rhumerus",
    "RightForeArm": "rradius",
    "RightHand": "rwrist",
    "RightFingerBase": "rwrist",
    "RightHandFinger1": "rwrist",
    "RThumb": "rwrist",
}


@dataclass(frozen=True)
class Rig:
    """Target rig in Y-up metres: bone names, parent indices (-1 for the root), rest heads and tails."""

    names: tuple[str, ...]
    parents: IntArray
    heads: FloatArray
    tails: FloatArray


@dataclass(frozen=True)
class Pose:
    """Per sample and target bone: world rotation from rest (samples, bones, 3, 3) and head position."""

    rotations: FloatArray
    heads: FloatArray


def _unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


def swing(source: FloatArray, target: FloatArray) -> FloatArray:
    """The smallest rotation taking direction `source` onto direction `target`."""
    a, b = _unit(source), _unit(target)
    axis = np.cross(a, b)
    sine, cosine = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if sine < 1e-9:
        if cosine < 0.0:
            raise ValueError("opposite rest directions have no unique swing")
        return np.eye(3)
    return Rotation.from_rotvec(axis / sine * np.arctan2(sine, cosine)).as_matrix()


def _rest_alignment(rig: Rig, skeleton: Skeleton) -> FloatArray:
    out = np.repeat(np.eye(3)[None], len(rig.names), axis=0)
    for i, name in enumerate(rig.names):
        source = CMU_MB[name]
        parent = int(rig.parents[i])
        if source == "root":
            continue
        if parent >= 0 and CMU_MB[rig.names[parent]] == source:
            out[i] = out[parent]
        else:
            out[i] = swing(rig.tails[i] - rig.heads[i], skeleton.bones[source].direction)
    return out


def _hip_height(skeleton: Skeleton) -> float:
    chain = ("lhipjoint", "lfemur", "ltibia")
    return float(-sum(skeleton.bones[b].direction[1] * skeleton.bones[b].length_m for b in chain))


def sample(motion: Motion, times: FloatArray) -> tuple[FloatArray, FloatArray]:
    """World rotations (samples, bones, 3, 3) and root positions at fractional source frames."""
    frames = np.arange(len(motion.rotations), dtype=np.float64)
    rotations = np.empty((len(times), len(motion.bones), 3, 3))
    for b in range(len(motion.bones)):
        slerp = Slerp(frames, Rotation.from_matrix(motion.rotations[:, b]))
        rotations[:, b] = slerp(times).as_matrix()
    root = np.stack([np.interp(times, frames, motion.heads[:, 0, k]) for k in range(3)], axis=-1)
    return rotations, root


def retarget(rig: Rig, skeleton: Skeleton, motion: Motion, times: FloatArray) -> Pose:
    rotations, root = sample(motion, times)
    align = _rest_alignment(rig, skeleton)
    index = [motion.bones.index(CMU_MB[name]) for name in rig.names]
    world = rotations[:, index] @ align[None]
    scale = _target_hip_height(rig) / _hip_height(skeleton)
    heads = np.zeros((len(times), len(rig.names), 3))
    for i in range(len(rig.names)):
        p = int(rig.parents[i])
        if p < 0:
            heads[:, i] = root * scale
            continue
        heads[:, i] = heads[:, p] + world[:, p] @ (rig.heads[i] - rig.heads[p])
    return Pose(world, heads)


def _target_hip_height(rig: Rig) -> float:
    hip = rig.heads[rig.names.index("LeftUpLeg")]
    ankle = rig.heads[rig.names.index("LeftFoot")]
    return float(hip[1] - ankle[1])


def skin(rest: FloatArray, joints: IntArray, weights: FloatArray, rig: Rig, pose: Pose) -> FloatArray:
    """Linear blend skinning of rest points (n, 3) for every pose sample: (samples, n, 3)."""
    out = np.zeros((len(pose.rotations), len(rest), 3))
    for k in range(joints.shape[1]):
        bone = joints[:, k]
        local = rest - rig.heads[bone]
        moved = np.einsum("snij,nj->sni", pose.rotations[:, bone], local) + pose.heads[:, bone]
        out += weights[None, :, k, None] * moved
    return out


def source_frames(start: int, frames: int, offsets: FloatArray, fps: float) -> FloatArray:
    """Source (120 fps) frame times for clip frames 0..frames-1 at each sub-frame offset."""
    clip = np.arange(frames, dtype=np.float64)[:, None] + offsets[None, :]
    return start + clip.ravel() * FPS / fps
