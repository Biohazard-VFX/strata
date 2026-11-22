"""Pixel roles shared by data generation and training, carried by linear deep channels, never by ids.

Every surface carries one constant primvar per role, 1 when it plays the role and 0 otherwise, chosen
by its `manifest.instance_ids` range name (`membership`): `foreground` is every range but the sky,
`hero` the hero's skin, groom and hairstyle, `fur` its groom and hairstyle strands. Each renders as a
deep channel premultiplied by the sample's float alpha, like `coverage`. Karma merges samples by
summing their channels, so a merged sample holds exactly the coverage of each role among its
sources; a role's pixel coverage sums its channel times the transmittance in front. The beauty
renders the same primvars as flat float AOVs (`AOVS`), an independent image of each role's coverage.
"""

from typing import Any, cast

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


def nested(raw: NDArray[np.floating], alpha: NDArray[np.float32]) -> NDArray[np.float32]:
    """Owned (N, 3) premultiplied sample role coverage, clamped to 0 <= fur <= hero <= foreground <= alpha.

    `raw` is the (N, 3) deep role channels in ROLES order, borrowed; float accumulation can leave
    them a few ulps outside that order.
    """
    foreground = np.clip(raw[:, 0], 0.0, alpha)
    hero = np.clip(raw[:, 1], 0.0, foreground)
    fur = np.clip(raw[:, 2], 0.0, hero)
    return np.ascontiguousarray(np.stack([foreground, hero, fur], axis=1), np.float32)


def sky(alpha: Any, foreground: Any) -> Any:
    """Samples mostly of the sky backdrop: their foreground share is below half their alpha."""
    return 2.0 * foreground < alpha


def coverage(
    alpha: NDArray[np.float32],
    weights: NDArray[np.float32],
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
    pixel = np.repeat(np.arange(counts.size), counts)
    out: dict[str, NDArray[np.float32]] = {}
    for index, channel in enumerate(CHANNELS):
        visible = weights[:, index].astype(np.float64) * transmitted
        summed = np.bincount(pixel, weights=visible, minlength=counts.size)
        out[channel] = np.clip(summed, 0.0, 1.0).astype(np.float32).reshape(shape)
    return out
