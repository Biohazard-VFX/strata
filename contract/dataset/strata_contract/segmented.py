"""Compiled segmented primitives; no fastmath or cross-pixel reductions."""

from importlib import import_module
from typing import Any

import numpy as np

jit: Any = import_module("numba").njit


@jit(cache=True)
def depth_order(depth: Any, counts: Any) -> Any:
    order = np.empty(depth.size, dtype=np.int64)
    first = 0
    for count in counts:
        for j in range(first, first + count):
            value = depth[j]
            k = j
            while k > first and depth[order[k - 1]] > value:
                order[k] = order[k - 1]
                k -= 1
            order[k] = j
        first += count
    return order


@jit(cache=True)
def transmittance(alpha: Any, counts: Any) -> Any:
    result = np.empty(alpha.size, dtype=np.float64)
    cumulative = 0.0
    first = 0
    for count in counts:
        origin = cumulative
        for j in range(first, first + count):
            before = cumulative
            a = min(max(np.float64(alpha[j]), 0.0), 1 - 1e-15)
            cumulative += np.log1p(-a)
            result[j] = np.exp(before - origin)
        first += count
    return result


@jit(cache=True)
def fill(
    output: Any,
    depth: Any,
    alpha: Any,
    rgb: Any,
    counts: Any,
    order: Any,
    trans: Any,
    logs: Any,
    sky: Any,
    layers: int,
    threshold: float,
) -> None:
    largest = int(counts.max()) if counts.size else 0
    kept = np.empty(largest, dtype=np.int64)
    gaps = np.empty(largest, dtype=np.float64)
    splits = np.empty(largest, dtype=np.bool_)
    first = 0
    for pixel in range(counts.size):
        length = 0
        for j in range(first, first + counts[pixel]):
            if trans[j] * alpha[j] > 1e-12:
                kept[length] = j
                gaps[length] = logs[order[j]] - logs[order[kept[length - 1]]] if length else -np.inf
                splits[length] = False
                length += 1
        first += counts[pixel]
        for _ in range(layers - 1):
            maximum = threshold
            chosen = -1
            for j in range(1, length):
                if gaps[j] > maximum:
                    maximum = gaps[j]
                    chosen = j
            if chosen >= 0:
                splits[chosen] = True
                gaps[chosen] = -np.inf
        start = 0
        slot = 0
        while start < length:
            end = start + 1
            while end < length and not splits[end]:
                end += 1
            visible = 0.0
            inverse = 0.0
            maximum_t = 0.0
            for k in range(start, end):
                j = kept[k]
                w = trans[j] * alpha[j]
                visible += w
                inverse += w / depth[order[j]]
                maximum_t = max(maximum_t, trans[j])
            index = pixel * layers + slot
            sky_coverage = 0.0
            if maximum_t > 1e-12:
                for k in range(start, end):
                    j = kept[k]
                    local = trans[j] / maximum_t
                    for channel in range(3):
                        output[index, channel + 2] += local * rgb[order[j], channel]
                    if sky[order[j]]:
                        sky_coverage += local * alpha[j]
                output[index, 1] = visible / maximum_t
            if visible > 1e-12 and np.float32(sky_coverage) <= np.float32(1e-12):
                output[index, 0] = inverse / visible
            if visible <= 1e-12:
                output[index] = 0
            start = end
            slot += 1
