"""CMU Acclaim motion capture (ASF skeleton + AMC motion) read into world-space bone rotations.

Units follow the ASF header: lengths divide by `length` to give inches. Output is metres, Y up,
the subject's left along +X and its rest-pose front along +Z. AMC frames are 120 per second.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from strata_datagen.usdutil import FloatArray

INCH_M = 0.0254
FPS = 120.0
ROOT_DOFS = ("tx", "ty", "tz", "rx", "ry", "rz")


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str
    direction: FloatArray
    length_m: float
    axis_deg: FloatArray
    dofs: tuple[str, ...]


@dataclass(frozen=True)
class Skeleton:
    bones: dict[str, Bone]
    order: tuple[str, ...]
    scale_m: float


@dataclass(frozen=True)
class Motion:
    """Per frame and bone (in skeleton order): world rotation (frames, bones, 3, 3), head and tail.

    A world rotation maps the rest pose to the posed bone, so a rest direction d becomes rotation @ d.
    """

    bones: tuple[str, ...]
    rotations: FloatArray
    heads: FloatArray
    tails: FloatArray

    def bone(self, name: str) -> int:
        return self.bones.index(name)


def _euler(degrees: FloatArray) -> FloatArray:
    return Rotation.from_euler("xyz", np.radians(degrees)).as_matrix()


def _blocks(text: str) -> list[dict[str, list[str]]]:
    blocks: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for raw in text.splitlines():
        words = raw.split()
        if not words:
            continue
        if words[0] == "begin":
            current = {}
        elif words[0] == "end":
            blocks.append(current)
        else:
            current[words[0]] = words[1:]
    return blocks


def read_asf(path: Path) -> Skeleton:
    text = path.read_text()
    header, rest = text.split(":bonedata", 1)
    bonedata, hierarchy = rest.split(":hierarchy", 1)
    units = header.split(":units", 1)[1].split()
    scale = INCH_M / float(units[units.index("length") + 1])
    parents: dict[str, str] = {}
    for line in hierarchy.splitlines():
        words = line.split()
        if words and words[0] not in ("begin", "end"):
            parents.update(dict.fromkeys(words[1:], words[0]))
    bones = {"root": Bone("root", "", np.zeros(3), 0.0, np.zeros(3), ROOT_DOFS)}
    for block in _blocks(bonedata):
        name = block["name"][0]
        bones[name] = Bone(
            name,
            parents[name],
            np.array([float(v) for v in block["direction"]]),
            float(block["length"][0]) * scale,
            np.array([float(v) for v in block["axis"][:3]]),
            tuple(block.get("dof", [])),
        )
    order: list[str] = ["root"]
    for name in order:
        order.extend(child for child, parent in parents.items() if parent == name)
    return Skeleton(bones, tuple(order), scale)


def read_amc(path: Path, skeleton: Skeleton) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Root translation in metres and per-bone Euler angles in degrees (x, y, z), one row per frame."""
    frames: list[dict[str, list[float]]] = []
    for line in path.read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith((":", "#")):
            continue
        if len(words) == 1 and words[0].isdigit():
            frames.append({})
        else:
            frames[-1][words[0]] = [float(v) for v in words[1:]]
    count = len(frames)
    angles = {name: np.zeros((count, 3)) for name in skeleton.order}
    root = np.zeros((count, 3))
    axes = {"rx": 0, "ry": 1, "rz": 2}
    for index, frame in enumerate(frames):
        for name, values in frame.items():
            for dof, value in zip(skeleton.bones[name].dofs, values, strict=True):
                if dof in axes:
                    angles[name][index, axes[dof]] = value
                else:
                    root[index, "xyz".index(dof[1])] = value * skeleton.scale_m
    return root, angles


def solve(skeleton: Skeleton, root: FloatArray, angles: dict[str, FloatArray]) -> Motion:
    """Forward kinematics: world rotation = parent world @ C @ R @ C^T, with C the bone's axis frame."""
    count, bones = len(root), len(skeleton.order)
    rotations = np.zeros((count, bones, 3, 3))
    heads = np.zeros((count, bones, 3))
    tails = np.zeros((count, bones, 3))
    for i, name in enumerate(skeleton.order):
        bone = skeleton.bones[name]
        axis = _euler(bone.axis_deg)
        local = axis @ _euler(angles[name]) @ axis.T
        if name == "root":
            rotations[:, i], heads[:, i], tails[:, i] = local, root, root
            continue
        p = skeleton.order.index(bone.parent)
        rotations[:, i] = rotations[:, p] @ local
        heads[:, i] = tails[:, p]
        tails[:, i] = heads[:, i] + rotations[:, i] @ (bone.direction * bone.length_m)
    return Motion(skeleton.order, rotations, heads, tails)


def load(asf: Path, amc: Path) -> tuple[Skeleton, Motion]:
    skeleton = read_asf(asf)
    root, angles = read_amc(amc, skeleton)
    return skeleton, solve(skeleton, root, angles)
