"""Deep sample identity: an integer instance id with an independent checksum channel.

Karma premultiplies every deep extra channel by the sample's float alpha and merges samples of one
pixel by summing them, so a merged sample holds the coverage-weighted mean of its sources' ids.
Such a mean can round to a declared id of a surface that is not there. `instance_check` carries
`check(id)`, a hash of the id in [1, 2), merged the same way: a mean of different ids' checks
lands within TOLERANCE of the check of the id it rounds to with probability about 4 * TOLERANCE.
"""

import numpy as np
from numpy.typing import NDArray

VERSION = "instance-check-1"
MAX_ID = 1 << 15
TOLERANCE = 2.0**-14
UNLABELLED = -1
_MASK = 0xFFFFFFFF


def check(ids: NDArray[np.integer]) -> NDArray[np.float32]:
    """h(id) in [1, 2): the murmur3 32-bit finalizer of id + golden ratio, top 23 bits as mantissa."""
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= MAX_ID):
        raise ValueError(f"instance ids must be in [0, {MAX_ID})")
    x = (ids.astype(np.uint64) + np.uint64(0x9E3779B9)) & np.uint64(_MASK)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(_MASK)
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(_MASK)
    x ^= x >> np.uint64(16)
    return (1.0 + (x >> np.uint64(9)).astype(np.float64) * 2.0**-23).astype(np.float32)


def decode(
    instance_id: NDArray[np.floating], instance_check: NDArray[np.floating], coverage: NDArray[np.floating]
) -> NDArray[np.int64]:
    """Per-sample ids; UNLABELLED where coverage, id domain or check fails.

    All three inputs are the deep channels as stored: premultiplied by the sample's float alpha.
    """
    alpha = coverage.astype(np.float64)
    positive = alpha > 0.0
    scale = np.where(positive, alpha, 1.0)
    quotient = np.where(positive, instance_id / scale, -1.0)
    ids = np.rint(np.clip(quotient, -1.0, MAX_ID)).astype(np.int64)
    known = positive & (ids >= 0) & (ids < MAX_ID)
    expected = check(np.where(known, ids, 0)).astype(np.float64)
    passed = known & (np.abs(instance_check / scale - expected) <= TOLERANCE)
    return np.where(passed, ids, UNLABELLED)


def range_errors(first_id: int, count: int) -> list[str]:
    """Why a manifest id range [first_id, first_id + count) cannot be decoded; empty when it can."""
    if first_id < 0 or count < 1 or first_id + count > MAX_ID:
        return [f"id range {first_id}+{count} leaves [0, {MAX_ID})"]
    return []
