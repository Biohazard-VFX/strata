"""Bounded segmented deep operations with stable source order and float64 sums."""

from importlib import import_module
from typing import Any

import numpy as np

jit: Any = import_module("numba").njit


@jit(cache=True)
def select_arrays(offsets: Any, z: Any, alpha: Any, rgb: Any, ids: Any, keep: Any) -> Any:
    size = int(keep.sum())
    selected_offsets = np.empty(offsets.size, dtype=np.int64)
    selected_z = np.empty(size, dtype=z.dtype)
    selected_alpha = np.empty(size, dtype=alpha.dtype)
    selected_rgb = np.empty((size, 3), dtype=rgb.dtype)
    selected_ids = np.empty(size, dtype=ids.dtype)
    out = 0
    selected_offsets[0] = 0
    for pixel in range(offsets.size - 1):
        for j in range(offsets[pixel], offsets[pixel + 1]):
            if keep[j]:
                selected_z[out] = z[j]
                selected_alpha[out] = alpha[j]
                selected_rgb[out] = rgb[j]
                selected_ids[out] = ids[j]
                out += 1
        selected_offsets[pixel + 1] = out
    return selected_offsets, selected_z, selected_alpha, selected_rgb, selected_ids


@jit(cache=True)
def geometry_bounds(offsets: Any, z: Any, ids: Any) -> Any:
    near = np.zeros(offsets.size - 1, dtype=z.dtype)
    far = np.zeros_like(near)
    valid = np.zeros(near.size, dtype=np.bool_)
    for pixel in range(near.size):
        for j in range(offsets[pixel], offsets[pixel + 1]):
            if ids[j] != 0:
                if not valid[pixel]:
                    near[pixel] = z[j]
                    valid[pixel] = True
                far[pixel] = z[j]
    return near, far, valid


@jit(cache=True)
def in_front(offsets: Any, z: Any, chosen: Any) -> Any:
    keep = np.empty(z.size, dtype=np.bool_)
    for pixel in range(offsets.size - 1):
        deepest = -np.inf
        for j in range(offsets[pixel], offsets[pixel + 1]):
            if chosen[j]:
                deepest = max(deepest, z[j])
        for j in range(offsets[pixel], offsets[pixel + 1]):
            keep[j] = z[j] < deepest
    return keep


@jit(cache=True)
def inserted_through(offsets: Any, alpha: Any, labels: Any, inserted: Any) -> Any:
    through = np.empty(offsets.size - 1, dtype=np.float64)
    for pixel in range(through.size):
        log_t = 0.0
        for j in range(offsets[pixel], offsets[pixel + 1]):
            if inserted[labels[j]]:
                log_t += np.log1p(-min(np.float64(alpha[j]), 1 - 1e-15))
        through[pixel] = np.exp(log_t)
    return through


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
