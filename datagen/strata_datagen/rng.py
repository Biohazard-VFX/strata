"""Named, independent random streams derived from one shot seed."""

import hashlib

import numpy as np


def stream(seed: int, *labels: str) -> np.random.Generator:
    """A generator that depends only on (seed, labels), so adding a stream never shifts another."""
    digest = hashlib.sha256(":".join((str(seed), *labels)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))
