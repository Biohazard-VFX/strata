"""Named surface regions on a human body mesh, for grooms and analysis: scalp, face, brows, beard.

Inputs are rest-pose vertex positions (Y up, facing +Z, metres), MPFB's scalp/ears/lips vertex
groups, the head bone weight per vertex and the eye centres. Regions are vertex masks; scalp, face
and body partition the skin; brows lie in the face, beard covers the lower face and throat.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from strata_datagen.usdutil import FloatArray, IntArray

Mask = NDArray[np.bool_]
HAIRLINE_ABOVE_EYES_M = 0.05
NAPE_BELOW_EYES_M = 0.075
BROW_BAND_M = (0.012, 0.035)
BROW_HALF_WIDTH_M = (0.012, 0.065)
BEARD_BELOW_EYES_M = 0.035


@dataclass(frozen=True)
class Landmarks:
    eye_left: FloatArray
    eye_right: FloatArray


def regions(
    points: FloatArray,
    head_weight: FloatArray,
    neck_weight: FloatArray,
    groups: dict[str, Mask],
    marks: Landmarks,
) -> dict[str, Mask]:
    eyes = 0.5 * (marks.eye_left + marks.eye_right)
    x, y, z = points[:, 0] - eyes[0], points[:, 1] - eyes[1], points[:, 2]
    head = head_weight >= 0.5
    ears = groups["ears"] & head
    ear_z = float(points[ears, 2].mean())
    back = z < ear_z - 0.01
    scalp = head & ~ears & (groups["scalp"] | (y > HAIRLINE_ABOVE_EYES_M) | (back & (y > -NAPE_BELOW_EYES_M)))
    face = head & ~scalp & ~ears & ~back
    side = np.abs(x)
    brows = (
        face
        & (y > BROW_BAND_M[0])
        & (y < BROW_BAND_M[1])
        & (side > BROW_HALF_WIDTH_M[0])
        & (side < BROW_HALF_WIDTH_M[1])
    )
    throat = (neck_weight >= 0.5) & (z > ear_z) & (y > -0.2)
    beard = ((face & (y < -BEARD_BELOW_EYES_M)) | throat) & ~groups["lips"]
    return {
        "scalp": scalp,
        "face": face,
        "brows": brows,
        "beard": beard,
        "ears": ears,
        "lips": groups["lips"] & face,
        "body": ~(scalp | face | ears),
    }


def face_subset(mask: Mask, counts: IntArray, indices: IntArray) -> IntArray:
    """Faces with at least half of their corners inside the vertex mask."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    inside = np.add.reduceat(mask[indices].astype(np.int64), starts)
    return np.flatnonzero(2 * inside >= counts).astype(np.int64)
