"""Bounded segmented deep operations with stable source order and float64 sums."""

from importlib import import_module
from typing import Any

import numpy as np

jit: Any = import_module("numba").njit


@jit(cache=True)
def merge_arrays(offsets: Any, zs: Any, alphas: Any, rgbs: Any, ids: Any) -> Any:
    pixels = offsets[0].size - 1
    groups = len(offsets)
    count = 0
    for z_values in zs:
        count += z_values.size
    result_offsets = np.zeros(pixels + 1, dtype=np.int64)
    z = np.empty(count, dtype=zs[0].dtype)
    alpha = np.empty(count, dtype=alphas[0].dtype)
    rgb = np.empty((count, 3), dtype=rgbs[0].dtype)
    identity = np.empty(count, dtype=ids[0].dtype)
    labels = np.empty(count, dtype=np.int64)
    cursors = np.zeros(groups, dtype=np.int64)
    out = 0
    for pixel in range(pixels):
        while True:
            nearest = np.inf
            group = -1
            for g in range(groups):
                j = cursors[g]
                if j < offsets[g][pixel + 1] and zs[g][j] < nearest:
                    nearest = zs[g][j]
                    group = g
            if group < 0:
                break
            j = cursors[group]
            z[out] = zs[group][j]
            alpha[out] = alphas[group][j]
            rgb[out] = rgbs[group][j]
            identity[out] = ids[group][j]
            labels[out] = group
            cursors[group] += 1
            out += 1
        result_offsets[pixel + 1] = out
    return result_offsets, z, alpha, rgb, identity, labels


@jit(cache=True)
def contributions_arrays(
    offsets: Any, alpha: Any, rgb: Any, labels: Any, visible: Any, groups: int
) -> Any:
    pixels = offsets.size - 1
    colour = np.zeros((groups, 3, pixels), dtype=np.float64)
    coverage = np.zeros((groups, pixels), dtype=np.float64)
    for pixel in range(pixels):
        for j in range(offsets[pixel], offsets[pixel + 1]):
            group = labels[j]
            coverage[group, pixel] += visible[j] * alpha[j]
            for channel in range(3):
                colour[group, channel, pixel] += visible[j] * rgb[j, channel]
    return colour, coverage
