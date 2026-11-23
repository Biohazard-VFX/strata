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
    zback: Any,
    statistics: Any,
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
            if statistics is not None:
                t = statistics[j, 0]
                weight = statistics[j, 1]
            if weight > 1e-12:
                log = np.log(np.float64(depth[j]))
                gaps[length] = log - previous_log if length else -np.inf
                previous_log = np.log(np.float64(zback[j]))
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
                if statistics is None:
                    inverse += weights[k] / depth[kept[k]]
                else:
                    inverse += statistics[kept[k], 2]
                maximum_t = max(maximum_t, trans[k])
            index = pixel * layers + slot
            sky_coverage = 0.0
            if maximum_t > 1e-12:
                for k in range(start, end):
                    j = kept[k]
                    local = trans[k] / maximum_t
                    for channel in range(3):
                        output[index, channel + 2] += local * rgb[j, channel]
                    if statistics is not None:
                        sky_coverage += statistics[j, 3] / maximum_t
                    elif sky[j]:
                        sky_coverage += local * np.float64(alpha[j])
                output[index, 1] = visible / maximum_t
            if visible > 1e-12 and np.float32(sky_coverage) <= np.float32(1e-12):
                output[index, 0] = inverse / visible
            if visible <= 1e-12:
                output[index] = 0
            start = end
            slot += 1


@jit(cache=True)
def merge_runs(
    depth: Any,
    alpha: Any,
    rgb: Any,
    roles: Any,
    ids: Any,
    counts: Any,
    sky: Any,
    threshold: float,
) -> Any:
    transmitted = transmittance(alpha, counts)
    starts = np.empty(depth.size, dtype=np.int64)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    first = 0
    runs = 0
    for pixel in range(counts.size):
        for j in range(first, first + counts[pixel]):
            join = False
            if j > first:
                join = (
                    ids[j] == ids[j - 1]
                    and np.log(np.float64(depth[j])) - np.log(np.float64(depth[j - 1])) <= threshold
                    and transmitted[j] * alpha[j] > 1e-12
                    and transmitted[j - 1] * alpha[j - 1] > 1e-12
                )
            if not join:
                starts[runs] = j
                runs += 1
        first += counts[pixel]
        offsets[pixel + 1] = runs
    z = np.empty(runs, dtype=np.float32)
    back = np.empty(runs, dtype=np.float32)
    coverage = np.zeros(runs, dtype=np.float64)
    color = np.zeros((runs, 3), dtype=np.float64)
    semantic = np.zeros((runs, 3), dtype=np.float64)
    labels = np.empty(runs, dtype=np.int64)
    statistics = np.zeros((runs, 4), dtype=np.float64)
    for run in range(runs):
        start = starts[run]
        end = starts[run + 1] if run + 1 < runs else depth.size
        z[run] = depth[start]
        back[run] = depth[end - 1]
        labels[run] = ids[start]
        statistics[run, 0] = transmitted[start]
        local = 1.0
        for j in range(start, end):
            a = np.float64(alpha[j])
            coverage[run] += local * a
            for channel in range(3):
                color[run, channel] += local * rgb[j, channel]
                semantic[run, channel] += local * roles[j, channel]
            weight = transmitted[j] * a
            statistics[run, 1] += weight
            statistics[run, 2] += weight / depth[j]
            if sky[j]:
                statistics[run, 3] += weight
            local *= 1.0 - a
    return offsets, z, back, coverage, color, semantic, labels, statistics
