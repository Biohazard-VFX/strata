"""Pixel roles shared by data generation and training: the visible coverage of named id ranges.

A role is a set of `manifest.instance_ids` range names, never fixed ids: `foreground` is every range
but the sky, `hero` the hero's skin, groom and hairstyle, `fur` its groom and hairstyle strands.
"""

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from strata_dataset import segmented

SKY = "sky"
FUR = ("hero_groom", "hero_hair")
HERO = ("hero_skin", *FUR)
ROLES = ("foreground", "hero", "fur")
CHANNELS = tuple(f"roles.{role}" for role in ROLES)

Ranges = dict[str, list[tuple[int, int]]]


def ranges(instance_ids: Sequence[Mapping[str, object]]) -> Ranges:
    """Each role's [first, end) id ranges, taken by name from a manifest's `instance_ids`."""
    named: dict[str, tuple[int, int]] = {}
    for record in instance_ids:
        first, count = cast("int", record["first_id"]), cast("int", record["count"])
        named[cast("str", record["name"])] = (first, first + count)
    return {
        "foreground": [span for name, span in named.items() if name != SKY],
        "hero": [named[name] for name in HERO if name in named],
        "fur": [named[name] for name in FUR if name in named],
    }


def coverage(
    alpha: NDArray[np.float32],
    ids: NDArray[np.int64],
    counts: NDArray[np.int64],
    shape: tuple[int, int],
    roles: Ranges,
) -> dict[str, NDArray[np.float32]]:
    """Owned (H, W) float32 coverage per channel name; inputs are borrowed and calls are thread-safe.

    Samples are pixel-major and front to back within a pixel; a role's coverage is the summed
    visible coverage (alpha times the transmittance in front) of its samples.
    """
    if alpha.shape != ids.shape or int(counts.sum()) != alpha.size or counts.size != shape[0] * shape[1]:
        raise ValueError("sample alpha and ids must match the sample counts and image shape")
    transmitted = cast("NDArray[np.float64]", segmented.transmittance(alpha, counts))
    visible = alpha.astype(np.float64) * transmitted
    pixel = np.repeat(np.arange(counts.size), counts)
    out: dict[str, NDArray[np.float32]] = {}
    for role, channel in zip(ROLES, CHANNELS, strict=True):
        member = np.zeros(ids.size, np.bool_)
        for low, high in roles[role]:
            member |= (ids >= low) & (ids < high)
        summed = np.bincount(pixel[member], weights=visible[member], minlength=counts.size)
        out[channel] = np.clip(summed, 0.0, 1.0).astype(np.float32).reshape(shape)
    return out
