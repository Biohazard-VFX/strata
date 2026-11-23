"""Pixel roles shared by data generation and training, carried by linear deep channels, never by ids.

Every surface carries one constant primvar per role, 1 when it plays the role and 0 otherwise, chosen
by its `manifest.instance_ids` range name (`membership`): `foreground` is every range but the sky,
`hero` the hero's skin, groom and hairstyle, `fur` its groom and hairstyle strands. Each renders as a
deep channel premultiplied by the sample's float alpha, like `coverage`. Karma merges samples by
summing their channels, so a merged sample holds exactly the coverage of each role among its
sources; a role's pixel coverage sums its channel times the transmittance in front. The beauty
renders the same primvars as flat float AOVs (`AOVS`), an independent image of each role's coverage.
"""

from typing import Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray

from strata_dataset import segmented

SKY = "sky"
FUR = ("hero_groom", "hero_hair")
HERO = ("hero_skin", *FUR)
ROLES = ("foreground", "hero", "fur")
CHANNELS = tuple(f"roles.{role}" for role in ROLES)
AOVS = tuple(f"{role}.{role}" for role in ROLES)


def membership(name: str) -> tuple[float, float, float]:
    """The constant value of each role primvar, in ROLES order, on surfaces of the range `name`."""
    return (float(name != SKY), float(name in HERO), float(name in FUR))


def nested(
    raw: NDArray[np.floating] | tuple[NDArray[np.floating], ...],
    alpha: NDArray[np.float32],
    *,
    order: NDArray[np.int64] | None = None,
) -> NDArray[np.float32]:
    """Owned nested roles; borrowed channels and alpha stay immutable and calls are thread-safe.

    Raw is (N, 3), or three planar channels. Optional order gathers raw channels, not the already
    ordered alpha. Float accumulation can leave rendered roles a few ulps outside nesting.
    """
    if isinstance(raw, np.ndarray) and (raw.ndim != 2 or raw.shape[1] != len(ROLES)):
        raise ValueError("raw roles must have three channels")
    channels = (raw[:, 0], raw[:, 1], raw[:, 2]) if isinstance(raw, np.ndarray) else raw
    if len(channels) != len(ROLES) or any(c.ndim != 1 or c.size != channels[0].size for c in channels):
        raise ValueError("raw role channels must be flat and equally sized")
    if order is None:
        if alpha.size != channels[0].size:
            raise ValueError("role channels must match sample alpha")
    else:
        segmented.validate_order(order, alpha.size, channels[0].size)
    channels = tuple(c.astype(np.float32) if c.dtype == np.float16 else c for c in channels)
    return _nested(channels, alpha, order)


@segmented.jit(cache=True)
def _nested(
    channels: tuple[NDArray[np.floating], ...],
    alpha: NDArray[np.float32],
    order: NDArray[np.int64] | None,
) -> NDArray[np.float32]:
    output = np.empty((alpha.size, 3), np.float32)
    for i in range(alpha.size):
        source = i if order is None else order[i]
        bound = np.float64(alpha[i])
        for channel in range(3):
            bound = np.minimum(np.maximum(np.float64(channels[channel][source]), 0.0), bound)
            output[i, channel] = bound
    return output


class _Coverage[Mask](Protocol):
    def __rmul__(self, other: float, /) -> Self: ...

    def __lt__(self, other: Self, /) -> Mask: ...


def sky[Mask](alpha: _Coverage[Mask], foreground: _Coverage[Mask]) -> Mask:
    """Samples mostly of the sky backdrop: their foreground share is below half their alpha."""
    return 2.0 * foreground < alpha


def coverage(
    alpha: NDArray[np.floating],
    weights: NDArray[np.floating],
    counts: NDArray[np.int64],
    shape: tuple[int, int],
) -> dict[str, NDArray[np.float32]]:
    """Owned (H, W) float32 coverage per channel name; inputs are borrowed and calls are thread-safe.

    Samples are pixel-major and front to back within a pixel; `weights` is their (N, 3) premultiplied
    role coverage in ROLES order. A role's coverage is the sum over samples of its weight times the
    transmittance in front.
    """
    if weights.shape != (alpha.size, len(ROLES)) or int(counts.sum()) != alpha.size:
        raise ValueError("sample alpha and role weights must match the sample counts")
    if counts.size != shape[0] * shape[1]:
        raise ValueError("sample counts must match the image shape")
    transmitted = cast("NDArray[np.float64]", segmented.transmittance(alpha, counts))
    summed = cast("NDArray[np.float64]", segmented.weighted_sum(transmitted, weights, counts))
    covered = np.clip(summed, 0.0, 1.0).astype(np.float32)
    return {
        channel: np.ascontiguousarray(covered[:, index].reshape(shape))
        for index, channel in enumerate(CHANNELS)
    }
