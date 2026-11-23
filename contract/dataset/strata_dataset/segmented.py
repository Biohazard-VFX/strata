"""Compiled segmented primitives with stable ordering and no fastmath."""

from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_order(order: NDArray[np.int64], size: int, source: int) -> None:
    if order.ndim != 1 or order.size != size or np.any(order < 0) or np.any(order >= source):
        raise ValueError("sample order must gather valid flat source indices")


jit: Any = import_module("numba").njit


@jit(cache=True)
def _sort(depth: Any, result: Any, first: int, count: int) -> None:
    for j in range(count):
        source = first + j
        value = depth[source]
        k = j
        while k > 0 and (depth[result[k - 1]] > value or np.isnan(depth[result[k - 1]])):
            if np.isnan(value):
                break
            result[k] = result[k - 1]
            k -= 1
        result[k] = source


@jit(cache=True)
def order(depth: Any, counts: Any) -> Any:
    result = np.empty(depth.size, dtype=np.int64)
    first = 0
    for count in counts:
        _sort(depth, result[first : first + count], first, count)
        first += count
    return result


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
def gather(channels: Any, indices: Any) -> Any:
    output = np.empty((indices.size, len(channels)), dtype=np.float32)
    for i in range(indices.size):
        for channel in range(len(channels)):
            output[i, channel] = channels[channel][indices[i]]
    return output


@jit(cache=True)
def weighted_sum(transmitted: Any, weights: Any, counts: Any) -> Any:
    output = np.zeros((counts.size, weights.shape[1]), dtype=np.float64)
    first = 0
    for pixel in range(counts.size):
        for j in range(first, first + counts[pixel]):
            for channel in range(weights.shape[1]):
                output[pixel, channel] += np.float64(weights[j, channel]) * transmitted[j]
        first += counts[pixel]
    return output


@jit(cache=True)
def fill(
    output: Any,
    depth: Any,
    alpha: Any,
    rgb: Any,
    counts: Any,
    sky: Any,
    layers: int,
    threshold: float,
) -> None:
    largest = int(counts.max()) if counts.size else 0
    order = np.empty(largest, dtype=np.int64)
    kept = np.empty(largest, dtype=np.int64)
    weights = np.empty(largest, dtype=np.float64)
    trans = np.empty(largest, dtype=np.float64)
    gaps = np.empty(largest, dtype=np.float64)
    splits = np.empty(largest, dtype=np.bool_)
    first = 0
    cumulative = 0.0
    for pixel in range(counts.size):
        count = counts[pixel]
        _sort(depth, order, first, count)
        length = 0
        origin = cumulative
        previous_log = 0.0
        for k in range(count):
            j = order[k]
            a = min(max(np.float64(alpha[j]), 0.0), 1.0)
            before = cumulative
            cumulative += np.log1p(-min(a, 1 - 1e-15))
            t = np.exp(before - origin)
            weight = t * a
            if weight > 1e-12:
                log = np.log(np.float64(depth[j]))
                gaps[length] = log - previous_log if length else -np.inf
                previous_log = log
                kept[length] = j
                trans[length] = t
                weights[length] = weight
                splits[length] = False
                length += 1
        first += count
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
                visible += weights[k]
                inverse += weights[k] / depth[kept[k]]
                maximum_t = max(maximum_t, trans[k])
            index = pixel * layers + slot
            sky_coverage = 0.0
            if maximum_t > 1e-12:
                for k in range(start, end):
                    j = kept[k]
                    local = trans[k] / maximum_t
                    for channel in range(3):
                        output[index, channel + 2] += local * rgb[j, channel]
                    if sky[j]:
                        sky_coverage += local * np.float64(alpha[j])
                output[index, 1] = visible / maximum_t
            if visible > 1e-12 and np.float32(sky_coverage) <= np.float32(1e-12):
                output[index, 0] = inverse / visible
            if visible <= 1e-12:
                output[index] = 0
            start = end
            slot += 1
