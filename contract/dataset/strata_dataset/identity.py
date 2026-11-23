"""Deep sample identity: a best-effort integer instance id for per-object use, never a role label.

Karma premultiplies every deep extra channel by the sample's float alpha and merges samples of one
pixel by summing them, so a merged sample holds the coverage-weighted mean of its sources' ids,
which can round to an absent id or to one that holds only part of the sample. A sample keeps its
rounded id r only if its quotient lies within max(RESIDUAL, RELATIVE_RESIDUAL x id) of r and both
`instance_check` and `instance_check2`, two hashes of the id in [1, 2) from different mixing
functions with different keys and merged the same way, lie within TOLERANCE of `check(r)` and
`check2(r)`.

Bound, conditional on a model: let m be the share of a merged sample held by sources whose id is not
r. Each merged check differs from r's by m x (the other sources' weighted mean check - r's check).
If r's two checks behave as independent uniform values given the other sources' checks and the
sources and shares do not depend on them, each passes with probability at most
p(m) = 2 TOLERANCE / m + 2^-23 (float grid), so a sample whose mislabelled share is at least m is
accepted with probability at most p(m)^2 (6e-8 at m = 1/2), for any number of sources and shares,
r present or absent, and the expected mislabelled share accepted per merged sample is at most
2 TOLERANCE = 2^-13. The hashes are fixed functions: this is a statistical model, checked
numerically on all ids, not a guarantee for any one chosen merge.
"""

from functools import cache

import numpy as np
from numpy.typing import NDArray

from strata_dataset import segmented

VERSION = "instance-check-3"
MAX_ID = 1 << 15
TOLERANCE = 2.0**-14
RESIDUAL = 0.05
RELATIVE_RESIDUAL = 2e-5
UNLABELLED = -1
_MASK = np.uint64(0xFFFFFFFF)
_KEYS = (np.uint64(0x9E3779B9), np.uint64(0x7F4A7C15))


def _mix(
    x: NDArray[np.uint64], multipliers: tuple[int, int], shifts: tuple[int, int, int]
) -> NDArray[np.uint64]:
    x = x & _MASK
    x ^= x >> np.uint64(shifts[0])
    x = (x * np.uint64(multipliers[0])) & _MASK
    x ^= x >> np.uint64(shifts[1])
    x = (x * np.uint64(multipliers[1])) & _MASK
    return x ^ (x >> np.uint64(shifts[2]))


def _keyed(ids: NDArray[np.integer], key: int) -> NDArray[np.uint64]:
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= MAX_ID):
        raise ValueError(f"instance ids must be in [0, {MAX_ID})")
    return ids.astype(np.uint64) + _KEYS[key]


def _unit(x: NDArray[np.uint64]) -> NDArray[np.float32]:
    return (1.0 + (x >> np.uint64(9)).astype(np.float64) * 2.0**-23).astype(np.float32)


def check(ids: NDArray[np.integer]) -> NDArray[np.float32]:
    """h(id) in [1, 2): the top 23 bits of the murmur3 32-bit finalizer of id + 0x9E3779B9 as mantissa."""
    return _unit(_mix(_keyed(ids, 0), (0x85EBCA6B, 0xC2B2AE35), (16, 13, 16)))


def check2(ids: NDArray[np.integer]) -> NDArray[np.float32]:
    """g(id) in [1, 2): the top 23 bits of the lowbias32 mix of id + 0x7F4A7C15, never of h's hash."""
    return _unit(_mix(_keyed(ids, 1), (0x7FEB352D, 0x846CA68B), (16, 15, 16)))


def decode(
    instance_id: NDArray[np.floating],
    instance_check: NDArray[np.floating],
    instance_check2: NDArray[np.floating],
    coverage: NDArray[np.floating],
    *,
    order: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Owned ids, optionally gathered in `order`; borrowed channels stay immutable and calls are thread-safe.

    UNLABELLED marks failed coverage, id domain, integer residual or either check. All four
    inputs are the stored deep channels, premultiplied by the sample's float alpha.
    """
    channels = np.broadcast_arrays(instance_id, instance_check, instance_check2, coverage)
    values = tuple(
        channel.astype(np.float32).ravel() if channel.dtype == np.float16 else channel.ravel()
        for channel in channels
    )
    if order is not None:
        segmented.validate_order(order, order.size, values[0].size)
    return _decode(values, _checks(), order).reshape(channels[0].shape if order is None else order.shape)


@cache
def _checks() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    ids = np.arange(MAX_ID, dtype=np.int64)
    return check(ids), check2(ids)


@segmented.jit(cache=True)
def _decode(
    channels: tuple[NDArray[np.floating], ...],
    checks: tuple[NDArray[np.float32], NDArray[np.float32]],
    order: NDArray[np.int64] | None,
) -> NDArray[np.int64]:
    raw, first, second, coverage = channels
    output = np.full(raw.size if order is None else order.size, UNLABELLED, np.int64)
    for i in range(output.size):
        source = i if order is None else order[i]
        alpha = np.float64(coverage[source])
        if alpha <= 0 or not np.isfinite(alpha):
            continue
        quotient = np.float64(raw[source]) / alpha
        if not np.isfinite(quotient):
            continue
        rounded = np.rint(min(max(quotient, -1.0), float(MAX_ID)))
        if rounded < 0 or rounded >= MAX_ID:
            continue
        decoded = int(rounded)
        if abs(quotient - decoded) > max(RESIDUAL, RELATIVE_RESIDUAL * abs(quotient)):
            continue
        one = abs(np.float64(first[source]) / alpha - np.float64(checks[0][decoded]))
        two = abs(np.float64(second[source]) / alpha - np.float64(checks[1][decoded]))
        if one <= TOLERANCE and two <= TOLERANCE:
            output[i] = decoded
    return output


def range_errors(first_id: int, count: int) -> list[str]:
    """Why a manifest id range [first_id, first_id + count) cannot be decoded; empty when it can."""
    if first_id < 0 or count < 1 or first_id + count > MAX_ID:
        return [f"id range {first_id}+{count} leaves [0, {MAX_ID})"]
    return []
