"""Deep sample identity: a best-effort integer instance id for per-object use, never a role label.

Karma premultiplies every deep extra channel by the sample's float alpha and merges samples of one
pixel by summing them, so a merged sample holds the coverage-weighted mean of its sources' ids,
which can round to an absent id or to one that holds only part of the sample. A sample keeps its
rounded id r only if its quotient lies within max(RESIDUAL, RELATIVE_RESIDUAL x id) of r and both
`instance_check` and `instance_check2`, two independent hashes of the id in [1, 2) merged the same
way, lie within TOLERANCE of `check(r)` and `check2(r)`.

Bound: let m be the share of a merged sample held by sources whose id is not r. Each merged check
differs from r's by m x (the other sources' weighted mean check - r's check). With r's two checks
independent and uniform given the other sources' checks, each passes with probability at most
p(m) = 2 TOLERANCE / m + 2^-23 (float grid), so a sample whose mislabelled share is at least m is
accepted with probability at most p(m)^2 (6e-8 at m = 1/2), for any number of sources and shares,
r present or absent. The expected mislabelled share accepted per merged sample is at most
2 TOLERANCE = 2^-13.
"""

import numpy as np
from numpy.typing import NDArray

VERSION = "instance-check-2"
MAX_ID = 1 << 15
TOLERANCE = 2.0**-14
RESIDUAL = 0.05
RELATIVE_RESIDUAL = 2e-5
UNLABELLED = -1
_MASK = np.uint64(0xFFFFFFFF)
_KEYS = (np.uint64(0x9E3779B9), np.uint64(0x7F4A7C15))


def _finalize(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    x = x & _MASK
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & _MASK
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & _MASK
    return x ^ (x >> np.uint64(16))


def _hash(ids: NDArray[np.integer]) -> NDArray[np.uint64]:
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= MAX_ID):
        raise ValueError(f"instance ids must be in [0, {MAX_ID})")
    return _finalize(ids.astype(np.uint64) + _KEYS[0])


def _unit(x: NDArray[np.uint64]) -> NDArray[np.float32]:
    return (1.0 + (x >> np.uint64(9)).astype(np.float64) * 2.0**-23).astype(np.float32)


def check(ids: NDArray[np.integer]) -> NDArray[np.float32]:
    """h(id) in [1, 2): the top 23 bits of the murmur3 32-bit finalizer of id + 0x9E3779B9 as mantissa."""
    return _unit(_hash(ids))


def check2(ids: NDArray[np.integer]) -> NDArray[np.float32]:
    """g(id) in [1, 2): the same finalizer applied again to that hash + 0x7F4A7C15, top 23 bits."""
    return _unit(_finalize(_hash(ids) + _KEYS[1]))


def decode(
    instance_id: NDArray[np.floating],
    instance_check: NDArray[np.floating],
    instance_check2: NDArray[np.floating],
    coverage: NDArray[np.floating],
) -> NDArray[np.int64]:
    """Per-sample ids; UNLABELLED where coverage, id domain, integer residual or either check fails.

    All four inputs are the deep channels as stored: premultiplied by the sample's float alpha.
    """
    alpha = coverage.astype(np.float64)
    positive = alpha > 0.0
    scale = np.where(positive, alpha, 1.0)
    quotient = np.where(positive, instance_id / scale, -1.0)
    ids = np.rint(np.clip(quotient, -1.0, MAX_ID)).astype(np.int64)
    integral = np.abs(quotient - ids) <= np.maximum(RESIDUAL, RELATIVE_RESIDUAL * np.abs(quotient))
    passed = positive & (ids >= 0) & (ids < MAX_ID) & integral
    safe = np.where(passed, ids, 0)
    for channel, hashed in ((instance_check, check), (instance_check2, check2)):
        passed &= np.abs(channel / scale - hashed(safe).astype(np.float64)) <= TOLERANCE
    return np.where(passed, ids, UNLABELLED)


def range_errors(first_id: int, count: int) -> list[str]:
    """Why a manifest id range [first_id, first_id + count) cannot be decoded; empty when it can."""
    if first_id < 0 or count < 1 or first_id + count > MAX_ID:
        return [f"id range {first_id}+{count} leaves [0, {MAX_ID})"]
    return []
