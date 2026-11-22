import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from strata_training.data.targets import reduce_samples
from strata_training.types import Array


def random_samples(seed: int) -> tuple[Array, Array, Array, Array, tuple[int, int]]:
    rng = np.random.default_rng(seed)
    shape = (3, 5)
    counts = rng.integers(0, 15, size=15)
    size = int(counts.sum())
    depth = rng.choice([1.0, 2.0, 4.0, 8.0, 16.0], size=size)
    alpha = rng.choice([0.0, 1e-13, 1e-12, 0.125, 0.5, 1.0], size=size)
    if seed % 2:
        depth = np.exp(rng.uniform(-3, 5, size=size))
        alpha[::3] = rng.uniform(0, 1, size=alpha[::3].size)
    rgb = rng.uniform(-2, 32, size=(size, 3)) * alpha[:, None]
    return depth, alpha, rgb, counts, shape


def test_textual_baseline_golden_is_bitwise_equal() -> None:
    path = Path(__file__).parent / "fixtures/reducer_30264d6.json"
    for case in json.loads(path.read_text()):
        result = reduce_samples(
            np.array(case["depth"]),
            np.array(case["alpha"]),
            np.array(case["rgb"]).reshape(-1, 3),
            np.array(case["counts"]),
            (2, 2),
            case["layers"],
        )
        for name, values in result.arrays().items():
            expected = np.array(case["expected"][name], dtype=np.float32)
            assert_array_equal(values, expected)
            assert values.tobytes() == expected.tobytes()


def test_randomized_baseline_fingerprints_are_bitwise_equal() -> None:
    path = Path(__file__).parent / "fixtures/reducer_random_30264d6.json"
    for case in json.loads(path.read_text()):
        result = reduce_samples(*random_samples(case["seed"]), layers=case["layers"])
        for name, values in result.arrays().items():
            assert hashlib.sha256(values.tobytes()).hexdigest() == case["sha256"][name], case
