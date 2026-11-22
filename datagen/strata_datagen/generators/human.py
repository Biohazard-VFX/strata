"""Place an animated hero clip (deforming meshes with their own looks) on the ground for one shot.

A clip is authored on flat ground at y = 0 with integer time codes 0..N+1, where 0 and N+1 are
motion-blur handles. A shot picks the clip frame shown at its first frame, a heading and a ground
position; the clip's root track then follows the terrain height frame by frame.
"""

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from strata_datagen.fields import Ground
from strata_datagen.usdutil import FloatArray, tag_instances

VERSION = "human-0.1.0"


@dataclass(frozen=True)
class ClipPlacement:
    """Clip frame shown first, heading, per-sample clip offset and ground point under the root.

    Samples cover the shot frames plus one handle each side, like the camera track.
    """

    start: int
    yaw_deg: float
    offset: FloatArray
    positions: FloatArray

    @property
    def speed_mps(self) -> float:
        steps = np.diff(self.positions[:, [0, 2]], axis=0)
        return float(np.linalg.norm(steps, axis=1).mean() * 24.0)


def _yaw(yaw_deg: float) -> FloatArray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def place(rng: np.random.Generator, ground: Ground, track: FloatArray, frames: int) -> ClipPlacement:
    """`track` holds the clip's root position per time code, handles included."""
    clip_frames = len(track) - 2
    start = int(rng.integers(1, clip_frames - frames + 2))
    yaw = float(rng.uniform(0.0, 360.0))
    window = track[start - 1 : start + frames + 1] @ _yaw(yaw).T
    centre = window[len(window) // 2]
    flat = window - np.array([centre[0], 0.0, centre[2]])
    lift = ground.height(flat[:, 0], flat[:, 2])
    offset = np.stack([np.full(len(flat), -centre[0]), lift, np.full(len(flat), -centre[2])], axis=-1)
    ground_points = np.stack([flat[:, 0], lift, flat[:, 2]], axis=-1)
    return ClipPlacement(start, yaw, offset, ground_points)


def author(
    stage: Usd.Stage, path: str, asset: str, placement: ClipPlacement, *, first_frame: int, instance_id: int
) -> UsdGeom.Xform:
    """Reference the clip so clip frame `start` plays at `first_frame`; its prims inherit the ids."""
    xform = UsdGeom.Xform(stage.DefinePrim(path))
    xform.GetPrim().GetReferences().AddReference(
        asset, layerOffset=Sdf.LayerOffset(float(first_frame - placement.start))
    )
    translate, rotate = xform.AddTranslateOp(), xform.AddRotateYOp()
    rotate.Set(placement.yaw_deg)
    for index, offset in enumerate(placement.offset):
        translate.Set(Gf.Vec3d(*(float(v) for v in offset)), Usd.TimeCode(first_frame - 1 + index))
    tag_instances(xform.GetPrim(), np.array([instance_id]))
    return xform
